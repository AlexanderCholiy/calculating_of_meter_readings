import json
import os
import time
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional, Union

import numpy as np
import pandas as pd

from calc.constants import (
    OUTPUT_CALC_FILE,
    ROUND_CALCULATION_DIGITS,
    ArchiveFile,
    IntegralReadingFile,
    PeriodReadingData,
    PeriodReadingFile,
)
from core.constants import DEBUG
from core.logger import calc_logger
from core.pretty_print import PrettyPrint
from core.wraps import retry
from db.connection import TSSessionLocal
from db.constants import (
    POLES_REPORT_CACHE_CACHE_FILE,
    POLES_REPORT_CACHE_TTL,
    PoleData,
)
from db.reports.poles_report import PoleReport

from .exceptions import ExcelSaveError, MissingColumnsError
from .services.algoritm import Algoritm


class MeterReadingsCalculator(
    IntegralReadingFile, ArchiveFile, PeriodReadingFile, Algoritm
):
    algoritm_column_name = 'Алгоритм'

    def __init__(self):
        self._poles_report: dict[str, PoleData] | None = None

    def _find_header_row(
        self,
        file_path: str,
        required_columns: tuple[str],
        max_rows: int = 25,
    ) -> int:
        """
        Ищет строку, содержащую все required_columns.
        Возвращает индекс строки для заголовков.
        """
        preview = pd.read_excel(
            file_path,
            header=None,
            nrows=max_rows,
        )

        required = set(required_columns)
        all_seen_columns = set()

        for idx, row in preview.iterrows():
            row_values = {str(cell).strip() for cell in row if pd.notna(cell)}
            if required.issubset(row_values):
                return idx

            all_seen_columns.update(row_values)

        missing = required - all_seen_columns
        actual_missing = tuple(missing) if missing else required_columns

        filename = os.path.basename(file_path)
        raise MissingColumnsError(actual_missing, filename)

    def _read_and_cast(
        self,
        file_path: str,
        required_columns: tuple[str],
        int_columns: tuple[str] = (),
        float_columns: tuple[str] = (),
        str_columns: tuple[str] = (),
        keep_columns: tuple[str] | None = None,
    ) -> pd.DataFrame:
        """
        Чтение Excel, поиск заголовков, проверка колонок и приведение типов.
        """
        header_row = self._find_header_row(
            file_path=file_path,
            required_columns=required_columns,
        )
        df = pd.read_excel(
            file_path,
            header=header_row,
        )
        df.columns = df.columns.astype(str).str.strip()

        for col in int_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        for col in float_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in str_columns:
            df[col] = df[col].astype('string')

        if keep_columns:
            df = df[[col for col in keep_columns if col in df.columns]]

        return df

    @property
    def integral_readings(self) -> pd.DataFrame:
        return self._read_and_cast(
            file_path=self.INTEGRAL_READINGS_FILE,
            required_columns=self.REQUIRED_INTEGRAL_READINGS_COLUMNS,
            int_columns=(
                self.CODE_EO_COL_IN_INTEGRAL_READINGS,
                self.PU_NUMBER_COL_IN_INTEGRAL_READINGS,
            ),
            float_columns=(
                self.PREV_READ_COL_IN_INTEGRAL_READINGS,
                self.CURRENT_READ_COL_IN_INTEGRAL_READINGS,
            ),
        )

    @property
    def archive(self) -> pd.DataFrame:
        return self._read_and_cast(
            file_path=self.ARCHIVE_FILE,
            required_columns=self.REQUIRED_ARCHIVE_COLUMNS,
            int_columns=(self.CODE_EO_COL_IN_ARCHIVE,),
            str_columns=(self.POLE_COL_IN_ARCHIVE,),
            keep_columns=self.REQUIRED_ARCHIVE_COLUMNS,
        )

    @property
    def period_readings(self) -> pd.DataFrame:
        return self._read_and_cast(
            file_path=self.PERIOD_READINGS_FILE,
            required_columns=self.REQUIRED_PERIOD_READINGS_COLUMNS,
            int_columns=(self.PU_NUMBER_COL_IN_PERIOD_READINGS,),
            str_columns=(
                self.POLE_COL_IN_PERIOD_READINGS,
                self.TU_TYPE_COL_IN_PERIOD_READINGS,
            ),
            float_columns=(
                self.LAST_READ_COL_IN_PERIOD_READINGS,
                self.EXP_COL_IN_PERIOD_READINGS,
            ),
            keep_columns=self.REQUIRED_PERIOD_READINGS_COLUMNS,
        )

    def _set_with_priority(
        self,
        storage: dict,
        key: tuple[str, int] | str,
        value: PeriodReadingData,
    ) -> bool:
        new_priority = self.TU_TYPE_PRIORITY.get(value['tu_type'], 0)

        old_value = storage.get(key)
        if old_value is None:
            return True

        old_priority = self.TU_TYPE_PRIORITY.get(old_value['tu_type'], 0)
        return new_priority > old_priority

    def get_period_readings_caches(
        self
    ) -> tuple[
        dict[tuple[str, int], PeriodReadingData], dict[str, PeriodReadingData]
    ]:
        """
        Кэш, для быстрого поиска значений из файла Показания за период.

        Возвращает два словаря для случаев, когда указан шифр опоры + номер ПУ,
        а также только шифр опоры.
        """
        result_with_pu_number = {}
        result_without_pu_number = {}

        period_readings = self.period_readings

        period_readings = period_readings.rename(columns={
            self.POLE_COL_IN_PERIOD_READINGS: 'pole',
            self.PU_NUMBER_COL_IN_PERIOD_READINGS: 'pu_number',
            self.TU_TYPE_COL_IN_PERIOD_READINGS: 'tu_type',
            self.LAST_READ_COL_IN_PERIOD_READINGS: 'last_read',
            self.START_DATE_COL_IN_PERIOD_READINGS: 'start_date',
            self.END_DATE_COL_IN_PERIOD_READINGS: 'end_date',
            self.EXP_COL_IN_PERIOD_READINGS: 'exp',
        })

        bad_key_counter = 0

        for row in period_readings.itertuples(index=False):
            pole: str = row.pole
            pu_number: np.int64 = row.pu_number

            if not isinstance(pole, str):
                bad_key_counter += 1
                continue

            tu_type: str = row.tu_type.lower().strip()
            last_read: Optional[float] = row.last_read
            exp: Optional[float] = row.exp

            start_date = None if pd.isna(row.start_date) else (
                row.start_date.to_pydatetime().replace(tzinfo=None)
            )

            end_date = None if pd.isna(row.end_date) else (
                row.end_date.to_pydatetime().replace(tzinfo=None)
            )

            value: PeriodReadingData = {
                'tu_type': tu_type,
                'last_read': last_read,
                'start_date': start_date,
                'end_date': end_date,
                'exp': exp,
            }

            # Важно сформировать два полных словаря:
            if not pd.isna(pu_number):
                base_key = (pole, int(pu_number))
                result_with_pu_number[base_key] = value

            add_key = pole
            if self._set_with_priority(
                result_without_pu_number, add_key, value
            ):
                result_without_pu_number[add_key] = value

        if bad_key_counter:
            filename = os.path.basename(self.PERIOD_READINGS_FILE)
            calc_logger.warning(
                f'В файле "{filename}" найдено {bad_key_counter} записей, '
                'у которых отсутствует значение в столбце '
                f'"{self.POLE_COL_IN_PERIOD_READINGS}".'
            )

        return result_with_pu_number, result_without_pu_number

    @property
    def poles_report(self):
        if self._poles_report is not None:
            return self._poles_report

        if POLES_REPORT_CACHE_CACHE_FILE.exists():
            file_age = (
                time.time()
                - POLES_REPORT_CACHE_CACHE_FILE.stat().st_mtime
            )

            if file_age < POLES_REPORT_CACHE_TTL:
                try:
                    with open(
                        POLES_REPORT_CACHE_CACHE_FILE, 'r', encoding='utf-8'
                    ) as f:
                        self._poles_report = json.load(f)

                    file_age_msg = PrettyPrint.format_seconds_2_human_time(
                        file_age
                    )
                    calc_logger.debug(
                        'Загружен отчет из файлового кэша '
                        f'(возраст: {file_age_msg})'
                    )
                    return self._poles_report

                except Exception as e:
                    calc_logger.warning(f'Ошибка чтения кэша: {e}. Идем в БД.')

        with TSSessionLocal() as session:
            self._poles_report = PoleReport.get_poles_with_master_flag(session)

        try:
            with open(
                POLES_REPORT_CACHE_CACHE_FILE, 'w', encoding='utf-8'
            ) as f:
                json.dump(self._poles_report, f, ensure_ascii=False, indent=4)
            calc_logger.debug(
                f'Отчет сохранен в кэш: {POLES_REPORT_CACHE_CACHE_FILE}'
            )
        except Exception as e:
            calc_logger.error(f'Не удалось сохранить кэш: {e}')

        return self._poles_report

    def calculations(self) -> None:
        """
        Дорасчитывает показания счётчиков с сохранением результатов в Excel.

        Также на отдельный лист сохраняем таблицу с:
            - Код ЭО
            - Шифр опоры
            - Номер ПУ
            - АСКУЭ
            - Расход (разница между текущими и предыдущими показаниями)
        """
        unique_archive = (
            self.archive
            .drop_duplicates(subset=[self.CODE_EO_COL_IN_ARCHIVE])
            .reset_index(drop=True)
        )
        new_integral_readings = self.integral_readings

        calc_data = new_integral_readings.merge(
            right=unique_archive,
            how='left',
            left_on=self.CODE_EO_COL_IN_INTEGRAL_READINGS,
            right_on=self.CODE_EO_COL_IN_ARCHIVE,
        )
        if (
            self.CODE_EO_COL_IN_INTEGRAL_READINGS
            != self.CODE_EO_COL_IN_ARCHIVE
        ):
            calc_data = calc_data.drop(columns=[self.CODE_EO_COL_IN_ARCHIVE])

        calc_data = calc_data.rename(columns={
            self.PU_NUMBER_COL_IN_INTEGRAL_READINGS: 'pu_number',
            self.POLE_COL_IN_ARCHIVE: 'pole',
            self.PREV_READ_COL_IN_INTEGRAL_READINGS: 'prev_readings',
            self.CURRENT_READ_COL_IN_INTEGRAL_READINGS: 'current_readings',
        })

        period_readings_with_pu_number, period_readings_without_pu_number = (
            self.get_period_readings_caches()
        )

        total = len(calc_data)

        meta = {
            'base_algoritm': 0,
            'add_algoritm': 0,
            'extra_algoritm': 0,
            'unknown_case': 0,
            'unvalid_case': 0,
            'total': total,
        }

        self.poles_report

        for row in calc_data.itertuples(index=True):
            idx: int = row.Index
            PrettyPrint.progress_bar_info(
                idx, total, 'Дорасчет интегральных показаний:'
            )

            pu_number: np.int64 = row.pu_number
            pole: Union[str, type(pd.NA)] = row.pole  # type: ignore
            prev_readings: float = row.prev_readings
            current_readings: Optional[float] = row.current_readings

            algoritm_name = 'unknown_case'

            if not pd.isna(current_readings):
                algoritm_name = 'unvalid_case'
                meta[algoritm_name] += 1
                calc_data.loc[idx, self.algoritm_column_name] = algoritm_name
                continue

            current_value = None

            # Алгоритмы расчёта:
            period_reading_data_with_pu_number = (
                period_readings_with_pu_number.get((pole, pu_number), None)
            ) if isinstance(pole, str) and not pd.isna(pu_number) else None

            period_reading_data_without_pu_number = (
                period_readings_without_pu_number.get(pole, None)
            ) if isinstance(pole, str) else None

            if period_reading_data_with_pu_number:
                current_value = self.base_algoritm(
                    period_reading_data_with_pu_number
                )
                if current_value is not None:
                    algoritm_name = 'base_algoritm'

            if current_value is None and period_reading_data_without_pu_number:
                current_value = self.add_algoritm(
                    period_reading_data_without_pu_number, prev_readings
                )
                if current_value is not None:
                    algoritm_name = 'add_algoritm'

            if current_value is None and not isinstance(pole, type(pd.NA)):
                current_value = self.extra_algoritm(
                    pole, self.poles_report, prev_readings
                )
                if current_value is not None:
                    algoritm_name = 'extra_algoritm'

            meta[algoritm_name] += 1

            # Дозаполняем текущие показания:
            if current_value is not None:
                new_integral_readings.loc[
                    idx, self.CURRENT_READ_COL_IN_INTEGRAL_READINGS
                ] = float(
                    Decimal(current_value)
                    .quantize(ROUND_CALCULATION_DIGITS, ROUND_HALF_UP)
                )

                calc_data.loc[idx, 'Расход'] = float(
                    Decimal(current_value - prev_readings)
                    .quantize(ROUND_CALCULATION_DIGITS, ROUND_HALF_UP)
                )

                calc_data.loc[idx, self.algoritm_column_name] = algoritm_name

        if meta['unknown_case']:
            calc_logger.warning(
                f'Найдено {meta['unknown_case']} / {total} записей, '
                'которые не удалось обработать текущими алгоритмами.'
            )

        calc_logger.debug(meta)

        self._save_calculations_results(new_integral_readings, calc_data)

    @retry(calc_logger, delay=30, exceptions=(ExcelSaveError,))
    def _save_calculations_results(
        self, new_integral_readings: pd.DataFrame, calc_data: pd.DataFrame
    ):
        for col in [self.CODE_EO_COL_IN_INTEGRAL_READINGS]:
            new_integral_readings[col] = (
                new_integral_readings[col].astype('string')
            )

        for col in [self.CODE_EO_COL_IN_INTEGRAL_READINGS]:
            calc_data[col] = (
                calc_data[col].astype('string')
            )

        try:
            with pd.ExcelWriter(
                OUTPUT_CALC_FILE, engine='openpyxl', mode='w'
            ) as writer:
                # Основной лист с расчетами:
                new_integral_readings.to_excel(
                    writer, sheet_name='calc', index=False
                )

                # Дополнительная информация:
                keep_columns = [
                    self.CODE_EO_COL_IN_INTEGRAL_READINGS,
                    'pole',
                    'pu_number',
                    self.ASKUE_COL_IN_ARCHIVE,
                    'Расход',
                ]

                if self.EXTRA_COLUMNS:
                    keep_columns.extend(self.EXTRA_COLUMNS)

                if DEBUG:
                    keep_columns.append(self.algoritm_column_name)

                add_info = calc_data[
                    [col for col in keep_columns if col in calc_data.columns]
                ].copy()

                add_info = add_info.rename(columns={
                    'pu_number': 'Номер ПУ',
                    'pole': 'Шифр опоры',
                })

                add_info.to_excel(writer, sheet_name='add', index=False)

                calc_logger.info(
                    'Результаты расчета интегральных показаний сохранены в '
                    f'{OUTPUT_CALC_FILE}'
                )

        except PermissionError:
            raise ExcelSaveError(
                file_path=OUTPUT_CALC_FILE,
                message=(
                    'Не удалось сохранить результаты дорасчёта в файл '
                    f'{OUTPUT_CALC_FILE} '
                    'Возможно, файл открыт в другой программе?'
                )
            )
