import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
from pandas import DataFrame

from core.constants import DEBUG
from core.logger import calc_logger
from core.pretty_print import PrettyPrint
from core.utils import clear_folder, get_sorted_excel_files, write_to_excel
from core.wraps import retry
from mail.constants import EMAIL_DIR
from mail.email_parser import email_config, email_parser

from .constants import (
    OUTPUT_POWER_PROFILE_CALC_FILE,
    POWER_CALC_RESULT_FILE,
    POWER_CALC_RESULT_TTL,
    PowerProfileFile
)
from .exceptions import EmptyPowerProfile, ExcelSaveError
from .power_calc import MeterReadingsCalculator
from .services.profile_algoritms import ProfileAlgoritm
from .services.profile_config import ConfigRestoreSignal


class DebugProfile(TypedDict):
    algoritm: str
    date: list[datetime]
    original_profile: np.ndarray
    restored_profile: np.ndarray


class PowerProfileCalc(PowerProfileFile, ProfileAlgoritm):

    algoritm_column_name = 'Алгоритм'

    _round_profile_key_digits: int = 2

    def __init__(self, scale_restored_only_val: bool = True, **kwargs):
        super().__init__(scale_restored_only_val, **kwargs)
        self._power_by_pole: dict[str, float] | None = None

    @property
    def power_by_pole(self):
        if POWER_CALC_RESULT_FILE.exists():
            file_age = (
                time.time()
                - POWER_CALC_RESULT_FILE.stat().st_mtime
            )

            if file_age < POWER_CALC_RESULT_TTL:
                try:
                    with open(
                        POWER_CALC_RESULT_FILE, 'r', encoding='utf-8'
                    ) as f:
                        self._power_by_pole = json.load(f)

                    file_age_msg = PrettyPrint.format_seconds_2_human_time(
                        file_age
                    )
                    calc_logger.debug(
                        'Загружены результаты расчета интегральных показаний '
                        f'из файлового кэша возраст: ({file_age_msg})'
                    )
                    return self._power_by_pole

                except (
                    PermissionError,
                    json.JSONDecodeError,
                    OSError,
                    UnicodeDecodeError
                ) as e:
                    calc_logger.warning(
                        f'Ошибка чтения кэша: {e}. '
                        'Запускаем расчет интегральных показаний.'
                    )

            calculator = MeterReadingsCalculator()
            self._power_by_pole = calculator.calculations()

            return self._power_by_pole

    def _get_debug_profiles_keys(self) -> set[str]:
        keys = []
        df = pd.read_excel(OUTPUT_POWER_PROFILE_CALC_FILE, 'graph')

        for row in df.itertuples(index=False):
            pole = row.pole
            pu_number = row.pu_number

            pole_val: Optional[str] = (
                str(pole).strip() if not pd.isna(pole) else None
            )
            pu_number_val: Optional[str] = (
                str(pu_number).strip() if not pd.isna(pu_number) else None
            )

            if pole_val and pu_number_val:
                key = (
                    MeterReadingsCalculator
                    .get_power_by_pole_key(pole_val, pu_number_val)
                )
                keys.append(key)

        return set(keys)

    def _get_date_columns(self, df: DataFrame) -> list[str]:
        date_columns = []

        for col in df.columns:
            parsed_date = pd.to_datetime(
                str(col),
                format=self.DATETIME_FORMAT_POWER_PROFILE,
                errors='coerce'
            )

            if pd.notna(parsed_date):
                date_columns.append(col)

        date_columns.sort(
            key=lambda x: pd.to_datetime(
                x, format=self.DATETIME_FORMAT_POWER_PROFILE
            )
        )

        return date_columns

    def get_power_profile(self, file: Path | str):
        df = MeterReadingsCalculator.read_and_cast_excel_file(
            file,
            # Только одна, т.к. табоицу надо потом транспонировать:
            required_columns=[self.POLE_COL_IN_POWER_PROFILE]
        )

        df = df.T

        df.columns = df.iloc[0]

        df = df.drop(df.index[0])

        df = df.reset_index()

        df = df.rename(
            columns={'index': self.POLE_COL_IN_POWER_PROFILE}
        )

        df.columns = list(df.columns)  # Переименование столбца index

        df.columns = df.columns.astype(str).str.strip()

        int_columns = [
            self.PU_NUMBER_COL_IN_POWER_PROFILE
        ]

        date_columns = self._get_date_columns(df)

        float_columns = [
            self.USE_BY_METER_COL_IN_POWER_PROFILE,
            self.USE_BY_PROFILE_COL_IN_POWER_PROFILE,
            self.DIF_COL_IN_POWER_PROFILE,
        ]
        float_columns.extend(date_columns)

        str_columns = [
            self.POLE_COL_IN_POWER_PROFILE,
            self.BS_COL_IN_POWER_PROFILE,
        ]

        keep_columns = str_columns.copy()
        keep_columns.extend(int_columns)
        keep_columns.extend(float_columns)

        for col in int_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        for col in float_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in str_columns:
            df[col] = df[col].astype('string')

        df = df[[col for col in keep_columns if col in df.columns]]

        df = df.copy()

        df['algoritm'] = None

        return df

    def _get_default_profile_key(self, power: float) -> float:
        """Безопасный ключ из float с защитой точности."""
        return round(power, self._round_profile_key_digits)

    def get_default_profiles(
        self, df: DataFrame
    ) -> dict[float, np.ndarray]:
        date_columns = self._get_date_columns(df)
        date_indices = [df.columns.get_loc(col) for col in date_columns]

        good_profiles: dict[float, np.ndarray] = {}

        for row in df.itertuples(index=False):
            y_vars = np.array(
                [row[j] for j in date_indices], dtype=float
            )
            use_by_readings = row.use_by_readings
            use_by_profile = row.use_by_profile

            if (
                pd.notna(use_by_readings)
                and pd.notna(use_by_profile)
                and not np.isnan(y_vars).any()
                and (y_vars > 0).all()
                and abs(use_by_readings - use_by_profile) <= 0.01
                and use_by_readings > 0
                and use_by_profile > 0
            ):
                key = self._get_default_profile_key(use_by_readings)
                if key not in good_profiles:
                    good_profiles[key] = y_vars
                if len(good_profiles) >= self.PROFILE_SUMPLES_NUMBERS:
                    break

        if len(good_profiles) < 1:
            raise ValueError(
                'В выгрузке не найдено ни одной записи с полными данными '
                'потребления, где показания по профилю совпадают с '
                'показаниями за период.'
            )

        return good_profiles

    def calculations(self, transpolation: bool = True):
        clear_folder(EMAIL_DIR)

        email_parser.parser(
            subject=email_config['EMAIL_SUBJECT'],
            days_before=email_config['EMAIL_DAYS_BEFORE'],
            sender_email=email_config['EMAIL_SENDER'],
        )

        power_map = self.power_by_pole

        calc_data: DataFrame = DataFrame()
        files = get_sorted_excel_files(EMAIL_DIR)

        for file in files:
            if not calc_data.empty:
                break

            calc_data = self.get_power_profile(file)

        if calc_data.empty:
            raise EmptyPowerProfile

        total = len(calc_data)

        calc_data = calc_data.rename(columns={
            self.POLE_COL_IN_POWER_PROFILE: 'pole',
            self.PU_NUMBER_COL_IN_POWER_PROFILE: 'pu_number',
            self.USE_BY_METER_COL_IN_POWER_PROFILE: 'use_by_readings',
            self.USE_BY_PROFILE_COL_IN_POWER_PROFILE: 'use_by_profile',
        })

        date_columns = self._get_date_columns(calc_data)
        date_indices = [calc_data.columns.get_loc(col) for col in date_columns]

        good_profiles = self.get_default_profiles(calc_data)
        good_profile_keys = list(good_profiles.keys())

        x_dt = [
            datetime.strptime(
                var, self.DATETIME_FORMAT_POWER_PROFILE
            ) for var in date_columns.copy()
        ]
        x_vars = self.datetime_to_seconds(x_dt)

        meta = {
            'full_filed': {
                'count': 0,
                'condition': 'Все данные за период заполнены'
            },
            'full_empty': {
                'count': 0,
                'condition': (
                    'Данные отсутствуют, '
                    'масштабируем случайный эталонный профиль'
                )
            },
            'periodic_internal_fill': {
                'count': 0,
                'condition': (
                    'Пропуски внутри ряда. Выполняется восстановление '
                    'периодической моделью с сохранением измеренных точек.'
                )
            },
            'periodic_edge_reconstruction': {
                'count': 0,
                'condition': (
                    'Пропуски на границах ряда. Выполняется восстановление '
                    'по среднесуточному профилю с масштабированием.'
                )
            },
            'periodic_hybrid_restore': {
                'count': 0,
                'condition': (
                    'Смешанный тип пропусков. Используется комбинированное '
                    'периодическое восстановление '
                    '(суточная модель + спектральное уточнение).'
                )
            },
            'unknown_case': {
                'count': 0,
                'condition': 'Не удалось применить ни один алгоритм '
                'восстановления данных'
            },
            'unvalid_case': {
                'count': 0,
                'condition': 'Некорректные данные (нет опоры или ПУ)'
            },
            'total': total,
        }

        algoritm_res = []
        use_by_readings_res = []
        use_by_profile_res = []
        dif_res = []
        date_values_res = []
        poles_res = []

        debug_profiles: dict[str, DebugProfile] = {}
        debug_profile_keys = (
            self._get_debug_profiles_keys() if DEBUG else set()
        )
        min_known_points_for_processing_full_empty = max(
            3, int(len(x_dt) * self.MIN_KNOWN_POINTS_FRACTION)
        )

        config_restore_signal = ConfigRestoreSignal(
            x_vars,
            noise_std_ratio=self.NOISE_STD_RATIO,
            min_block_ratio=self.MIN_BLOCK_RATIO,
        )

        for i, row in enumerate(calc_data.itertuples(index=False)):
            PrettyPrint.progress_bar_info(
                i, total, 'Дорасчет профилей мощности:'
            )

            algoritm_name = None

            pole = row.pole.strip().split('.')[0] if row.pole else None

            pu_number = row.pu_number

            y_vars = np.array(
                [row[j] for j in date_indices],
                dtype=float
            )
            y_vars_original = y_vars.copy()

            mask = np.isnan(y_vars)

            power_key = MeterReadingsCalculator.get_power_by_pole_key(
                pole, pu_number
            )

            # Приоритет у недавно посчитанного профиля:
            total_power = pd.NA
            power_map_value: Optional[float] = power_map.get(power_key)

            if power_map_value and power_map_value > 0:
                total_power = power_map_value
            elif not pd.isna(row.use_by_profile) and row.use_by_profile > 0:
                total_power = row.use_by_profile
            elif not pd.isna(row.use_by_readings) and row.use_by_readings > 0:
                total_power = row.use_by_readings

            if pd.isna(pu_number) or not pole:
                algoritm_name = 'unvalid_case'
                meta[algoritm_name]['count'] += 1
                total_power = pd.NA

            elif not mask.any():
                algoritm_name = 'full_filed'
                meta[algoritm_name]['count'] += 1
                is_not_all_zero = (y_vars > 0).all()

                if (
                    not self.scale_restored_only_val
                    and not pd.isna(total_power)
                    and total_power > 0
                    and is_not_all_zero
                ):
                    y_vars = self.scale_to_area(y_vars, total_power)

                if (
                    not pd.isna(row.use_by_readings)
                    and not pd.isna(row.use_by_profile)
                    and abs(
                        row.use_by_readings - row.use_by_profile
                    ) <= 0.01
                    and is_not_all_zero
                    and row.use_by_readings > 0
                    and row.use_by_profile > 0
                ):
                    key = self._get_default_profile_key(row.use_by_readings)
                    if key not in good_profiles:
                        good_profiles[key] = y_vars
                        good_profile_keys.append(key)

            elif not pd.isna(total_power) and total_power > 0:
                first_valid = np.argmax(~mask)
                last_valid = len(mask) - np.argmax(~mask[::-1]) - 1

                has_left_gap = first_valid > 0
                has_right_gap = last_valid < len(mask) - 1
                has_inside_gap = mask[first_valid:last_valid + 1].any()

                if (
                    not np.isfinite(y_vars).any()
                    or np.count_nonzero(np.isfinite(y_vars)) < (
                        min_known_points_for_processing_full_empty
                    )
                ):
                    y_vars = self.full_empty_algoritm(
                        good_profiles, good_profile_keys, total_power
                    )
                    algoritm_name = 'full_empty'
                    meta[algoritm_name]['count'] += 1
                elif has_inside_gap and not (has_left_gap or has_right_gap):
                    y_vars = self.restore_periodic_signal_algoritm(
                        x_vars, y_vars, total_power, config_restore_signal
                    )
                    algoritm_name = 'periodic_internal_fill'
                    meta[algoritm_name]['count'] += 1
                elif (has_left_gap or has_right_gap) and not has_inside_gap:
                    y_vars = self.restore_periodic_signal_algoritm(
                        x_vars, y_vars, total_power, config_restore_signal
                    )
                    algoritm_name = 'periodic_edge_reconstruction'
                    meta[algoritm_name]['count'] += 1
                else:
                    y_vars = self.restore_periodic_signal_algoritm(
                        x_vars, y_vars, total_power, config_restore_signal
                    )
                    algoritm_name = 'periodic_hybrid_restore'
                    meta[algoritm_name]['count'] += 1

            else:
                algoritm_name = 'unknown_case'
                meta[algoritm_name]['count'] += 1

            use_by_readings = (
                y_vars.sum()
                if np.isfinite(y_vars).any() else row.use_by_readings
            )
            use_by_profile = total_power

            # округление и нормализация значений дат
            if algoritm_name in (
                'full_empty',
                'periodic_internal_fill',
                'periodic_edge_reconstruction',
                'periodic_hybrid_restore',
            ):
                round_y_vars = np.round(y_vars, 5)
            else:
                round_y_vars = np.round(y_vars, 3)

            date_values_res.append(round_y_vars)

            use_by_readings_res.append(
                MeterReadingsCalculator.round_decimal(use_by_readings)
            )
            use_by_profile_res.append(
                MeterReadingsCalculator.round_decimal(use_by_profile)
            )

            delta = pd.NA

            if (
                not pd.isna(use_by_readings)
                and not pd.isna(use_by_profile)
                and use_by_readings is not None
                and use_by_profile is not None
            ):
                delta = MeterReadingsCalculator.round_decimal(
                    use_by_readings - use_by_profile
                )

            dif_res.append(delta)

            algoritm_res.append(algoritm_name)
            poles_res.append(pole)

            if power_key in debug_profile_keys:
                debug_profiles[power_key] = {
                    'algoritm': algoritm_name,
                    'date': x_dt,
                    'original_profile': y_vars_original,
                    'restored_profile': y_vars,
                }

        calc_data['algoritm'] = algoritm_res
        calc_data['use_by_readings'] = use_by_readings_res
        calc_data['use_by_profile'] = use_by_profile_res
        calc_data[self.DIF_COL_IN_POWER_PROFILE] = dif_res
        calc_data[date_columns] = date_values_res
        calc_data['pole'] = poles_res

        if meta['unknown_case']:
            calc_logger.warning(
                f'Найдено {meta["unknown_case"]["count"]} / {total} записей, '
                'которые не удалось обработать текущими алгоритмами.'
            )

        calc_logger.debug(meta)

        self._save_calculations_results_2_excel(calc_data, transpolation)

        if DEBUG:
            self._save_debug_profiles_to_excel(debug_profiles)

    @retry(calc_logger, delay=30, exceptions=(ExcelSaveError,))
    def _save_calculations_results_2_excel(
        self, calc_data: DataFrame, transpolation: bool
    ):
        calc_data = calc_data.rename(columns={
            'pole': self.POLE_COL_IN_POWER_PROFILE,
            'pu_number': self.PU_NUMBER_COL_IN_POWER_PROFILE,
            'use_by_readings': self.USE_BY_METER_COL_IN_POWER_PROFILE,
            'use_by_profile': self.USE_BY_PROFILE_COL_IN_POWER_PROFILE,
            'algoritm': self.algoritm_column_name,
        })

        keep_columns = list(calc_data.columns)

        if DEBUG:
            keep_columns.remove(self.algoritm_column_name)

            insert_pos = keep_columns.index(self.DIF_COL_IN_POWER_PROFILE) + 1

            keep_columns.insert(insert_pos, self.algoritm_column_name)
        else:
            keep_columns.remove(self.algoritm_column_name)

        df_to_save = calc_data[keep_columns]

        if transpolation:
            df_to_save = df_to_save.T
            to_excel_kwargs = {
                'index': True,
                'header': False,
            }
        else:
            to_excel_kwargs = {'index': False}

        write_to_excel(
            OUTPUT_POWER_PROFILE_CALC_FILE,
            'profile', df_to_save, **to_excel_kwargs
        )

    @retry(calc_logger, delay=30, exceptions=(ExcelSaveError,))
    def _save_debug_profiles_to_excel(
        self, debug_profiles: dict[str, DebugProfile]
    ):
        rows = []

        sorted_items = sorted(
            debug_profiles.items(),
            key=lambda item: (item[1]['algoritm'], item[0])
        )

        for key, profile in sorted_items:
            pole, pu = key.split('__', 1)

            base = {
                'pole': pole,
                'pu_number': pu,
                'algoritm': profile['algoritm'],
            }

            rows.append({
                'type': 'original',
                **base,
                'algoritm': pd.NA,
                **dict(zip(profile['date'], profile['original_profile']))
            })

            rows.append({
                'type': 'restored',
                **base,
                **dict(zip(profile['date'], profile['restored_profile']))
            })

        df = pd.DataFrame(rows)

        to_excel_kwargs = {'index': False}

        write_to_excel(
            OUTPUT_POWER_PROFILE_CALC_FILE,
            'debug', df, **to_excel_kwargs
        )
