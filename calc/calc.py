import os
import numpy as np
from typing import Union, Optional

import pandas as pd

from .constants import (
    IntegralReadingFile,
    ArchiveFile,
    PeriodReadingFile,
    OUTPUT_CALC_FILE,
)
from core.logger import calc_logger
from .exceptions import MissingColumnsError, ExcelSaveError
from core.wraps import retry


class MeterReadingsCalculator(
    IntegralReadingFile, ArchiveFile, PeriodReadingFile
):

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

        for idx, row in preview.iterrows():
            row_values = {str(cell).strip() for cell in row if pd.notna(cell)}
            if required.issubset(row_values):
                return idx

        filename = os.path.basename(file_path)
        raise MissingColumnsError(required_columns, filename)

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

        for row in calc_data.itertuples(index=True):
            idx: int = row.Index

            pu_number: np.int64 = row.pu_number
            pole: Union[str, type(pd.NA)] = row.pole  # type: ignore
            prev_readings: float = row.prev_readings
            current_readings: Optional[float] = row.current_readings

            if not pd.isna(current_readings):
                continue

            # Дозаполняем текущие показания:
            current_value = prev_readings

            new_integral_readings.loc[
                idx, self.CURRENT_READ_COL_IN_INTEGRAL_READINGS
            ] = current_value

            # Расход:
            calc_data.loc[idx, 'Расход'] = current_value - prev_readings

        self._save_calculations_results(new_integral_readings, calc_data)

    @retry(calc_logger, delay=30, exceptions=(ExcelSaveError,))
    def _save_calculations_results(
        self, new_integral_readings: pd.DataFrame, calc_data: pd.DataFrame
    ):
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
                add_info = calc_data[
                    [col for col in keep_columns if col in calc_data.columns]
                ].copy()

                add_info = add_info.rename(columns={
                    'pu_number': 'Номер ПУ',
                    'pole': 'Шифр опоры',
                })

                add_info.to_excel(writer, sheet_name='add', index=False)

        except PermissionError:
            raise ExcelSaveError(
                file_path=OUTPUT_CALC_FILE,
                message=(
                    'Не удалось сохранить результаты дорасчёта в файл '
                    f'{OUTPUT_CALC_FILE} '
                    'Возможно, файл открыт в другой программе?'
                )
            )
