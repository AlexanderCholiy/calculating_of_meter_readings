import time
import json
import pandas as pd
from pandas import DataFrame
from typing import Optional
from pathlib import Path
from datetime import datetime

from .power_calc import MeterReadingsCalculator
from .constants import POWER_CALC_RESULT_FILE, POWER_CALC_RESULT_TTL
from core.pretty_print import PrettyPrint
from core.logger import calc_logger
from mail.constants import EMAIL_DIR, FILENAME_DATETIME_PREFIX

from mail.email_parser import email_parser, email_config
from .exceptions import EmptyPowerProfile
from core.utils import get_sorted_excel_files
from .power_calc import MeterReadingsCalculator
from .constants import PowerProfileFile, PROFILE_SUMPLES_NUMBERS
from .services.profile_algoritms import ProfileAlgoritm


class PowerProfileCalc(PowerProfileFile, ProfileAlgoritm):

    def __init__(self):
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

                except Exception as e:
                    calc_logger.warning(
                        f'Ошибка чтения кэша: {e}. '
                        'Запускаем расчет интегральных показаний.'
                    )

            calculator = MeterReadingsCalculator()
            self._power_by_pole = calculator.calculations()

            return self._power_by_pole

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

        return df

    def calculations(self):
        # email_parser.parser(
        #     subject=email_config['EMAIL_SUBJECT'],
        #     days_before=email_config['EMAIL_DAYS_BEFORE'],
        #     sender_email=email_config['EMAIL_SENDER'],
        # )

        power_by_pole = self.power_by_pole

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

        x_vars = [
            datetime.strptime(
                var, self.DATETIME_FORMAT_POWER_PROFILE
            ) for var in date_columns.copy()
        ]

        meta = {
            'full_empty_algoritm': 0,
            'total': total,
        }

        good_profiles: dict[str, list[float]] = {}

        for row in calc_data.itertuples(index=True):
            y_vars = [
                round(row[i], 3)
                if not pd.isna(row[i]) else None for i in date_indices
            ]

            use_by_readings = row.use_by_readings
            use_by_profile = row.use_by_profile

            if (
                pd.isna(use_by_readings)
                or pd.isna(use_by_profile)
                or None in y_vars
                or abs(use_by_readings - use_by_profile) > 0.01
            ):
                continue

            if len(good_profiles) > PROFILE_SUMPLES_NUMBERS:
                break

            good_profiles[use_by_readings] = y_vars

        if len(good_profiles) < 1:
            raise ValueError(
                'В выгрузке не найдено ни одной записи с полными данными '
                'потребления, где показания по профилю совпадают с '
                'показаниями за период.'
            )

        good_profile_keys = list(good_profiles.keys())

        for row in calc_data.itertuples(index=True):
            idx: int = row.Index
            PrettyPrint.progress_bar_info(
                idx, total, 'Дорасчет профилей мощности:'
            )

            pole: Optional[str] = row.pole.strip() if row.pole else None
            pu_number: int = row.pu_number

            if pd.isna(pu_number) or not pole:
                continue

            y_vars = [
                row[i] if not pd.isna(row[i]) else None for i in date_indices
            ]

            if None not in y_vars:
                continue

            self.full_empty_algoritm(y_vars, good_profiles, good_profile_keys)
