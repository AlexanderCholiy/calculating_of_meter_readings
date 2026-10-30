import calendar
import os

import pandas as pd

from calc.constants import (
    PeriodReadingData,
    PeriodReadingFile,
)
from calc.exceptions import NotMasterPole
from calc.services.is_boundary_period import is_boundary_period
from db.connection import TSSessionLocal
from db.constants import PoleData
from db.models import Pole


class Algoritm:

    _poles_networks: dict[str, set[str]] | None = None

    def base_algoritm(
        self,
        period_reading_data: PeriodReadingData,
        pole: str,
        poles_report: dict[str, PoleData],
    ) -> float | None:
        """
        Применяем данный алгоритм, если "Номер ПУ" и "Шифр опоры" такие же, как
        в файле с Показаниями за период. Если опора питается от другой опоры
        сразу возвращаем 0.
        """
        start_date = period_reading_data['start_date']
        end_date = period_reading_data['end_date']
        last_read = period_reading_data['last_read']
        exp = period_reading_data['exp']

        pole_data = poles_report.get(pole)

        if pole_data is None:
            raise ValueError(
                f'Опоры {pole} не существует. Проверьте данные.'
            )

        power_source_pole = pole_data['power_source_pole']

        if power_source_pole:
            return last_read if not pd.isna(last_read) else None

        if any(
            v is None or pd.isna(v) for v in (start_date, end_date, last_read)
        ):
            return

        delta_days = (end_date - start_date).days

        if delta_days == 0:
            return

        filename = os.path.basename(PeriodReadingFile.PERIOD_READINGS_FILE)

        if start_date > end_date:
            raise ValueError(
                f'В файле "{filename}" найдена запись, где дата начала больше '
                'даты окончания.'
            )

        days_in_month = calendar.monthrange(
            start_date.year,
            start_date.month
        )[1]

        check_days = days_in_month + 1

        if delta_days > check_days:
            raise ValueError(
                f'В файле "{filename}" найдена запись, где разница между '
                f'{PeriodReadingFile.END_DATE_COL_IN_PERIOD_READINGS} и '
                f'{PeriodReadingFile.START_DATE_COL_IN_PERIOD_READINGS} '
                f'превышает {check_days} день. '
                'Пожалуйста, проверьте корректность данных.'
            )

        if is_boundary_period(start_date, end_date):
            return last_read

        if exp is None or pd.isna(exp):
            return

        return (
            exp
            * (1 + (days_in_month - delta_days) / delta_days)
            + last_read
        )

    def add_algoritm(
        self,
        period_reading_data: PeriodReadingData,
        prev_readings: float,
        pole: str,
        poles_report: dict[str, PoleData],
        transformation_factor: float = 1,
    ) -> float | None:
        """
        Применяем данный алгоритм, если "Номер ПУ" отсутствует, а "Шифр опоры"
        такой же, как в файле с Показаниями за период.
        Только для типа счетчика: "Комерческий", "Контрольный"
        """
        start_date = period_reading_data['start_date']
        end_date = period_reading_data['end_date']
        exp = period_reading_data['exp']

        tu_type = period_reading_data['tu_type']

        if tu_type not in PeriodReadingFile.TU_TYPE_PRIORITY:
            if tu_type in PeriodReadingFile.ADD_TU_TYPE:
                return

            valid_tu_types = sorted(
                list(PeriodReadingFile.ADD_TU_TYPE)
                + list(PeriodReadingFile.TU_TYPE_PRIORITY.keys())
            )

            raise ValueError(
                f'Проверьте тип точки учета: {tu_type}. '
                f'Возможные значения: {", ".join(valid_tu_types)}'
            )

        pole_data = poles_report.get(pole)

        if pole_data is None:
            raise ValueError(
                f'Опоры {pole} не существует. Проверьте данные.'
            )

        power_source_pole = pole_data['power_source_pole']
        if power_source_pole:
            return prev_readings

        if any(
            v is None or pd.isna(v) for v in (start_date, end_date, exp)
        ):
            return

        delta_days = (end_date - start_date).days

        if delta_days == 0:
            return

        filename = os.path.basename(PeriodReadingFile.PERIOD_READINGS_FILE)

        if start_date > end_date:
            raise ValueError(
                f'В файле "{filename}" найдена запись, где дата начала больше '
                'даты окончания.'
            )

        days_in_month = calendar.monthrange(
            start_date.year,
            start_date.month
        )[1]

        check_days = days_in_month + 1

        if delta_days > check_days:
            raise ValueError(
                f'В файле "{filename}" найдена запись, где разница между '
                f'{PeriodReadingFile.END_DATE_COL_IN_PERIOD_READINGS} и '
                f'{PeriodReadingFile.START_DATE_COL_IN_PERIOD_READINGS} '
                f'превышает {check_days} день. '
                'Пожалуйста, проверьте корректность данных.'
            )

        _transformation_factor = (
            transformation_factor
            if tu_type in PeriodReadingFile.CONTROL_TU_TYPES_PRIORITY
            else 1
        )

        if is_boundary_period(start_date, end_date):
            return exp / _transformation_factor + prev_readings

        return (
            exp / _transformation_factor
            * (1 + (days_in_month - delta_days) / delta_days)
            + prev_readings
        )

    def extra_algoritm(
        self,
        pole: str,
        poles_report: dict[str, PoleData],
        prev_readings: float,
        avg_exp: float,
        default_operator_group_count: float | None,
        transformation_factor: float = 1,
    ) -> float | None:
        """
        Применяем данный алгоритм в последнем случае, как расчет по среднему.
        """
        pole_data = poles_report.get(pole)

        if pole_data is None:
            return

        power_source_pole = pole_data['power_source_pole']
        is_master = pole_data['is_master']
        is_standalone = pole_data['is_standalone']
        operator_group_count = (
            default_operator_group_count
            if default_operator_group_count is not None
            else pole_data['operator_group_count']
        )

        if power_source_pole:
            return prev_readings

        if is_standalone:
            return (
                operator_group_count * avg_exp / transformation_factor
                + prev_readings
            )

        if is_master:
            all_operator_group_count = self.get_total_operator_count(
                pole, poles_report
            )
            return (
                all_operator_group_count * avg_exp / transformation_factor
                + prev_readings
            )

    def _get_poles_networks(self) -> dict[str, set[str]]:
        """
        Возвращает словарь:
        {
            'MASTER_POLE': {'SUB_POLE_1', 'SUB_POLE_2'}
        }
        """
        if self._poles_networks is not None:
            return self._poles_networks

        with TSSessionLocal() as session:
            rows = (
                session
                .query(Pole.pole, Pole.power_source_pole)
                .filter(Pole.power_source_pole.isnot(None))
                .all()
            )

        networks: dict[str, set[str]] = {}

        for sub_pole, master_pole in rows:
            networks.setdefault(master_pole, set()).add(sub_pole)

        for master, subs in networks.items():
            subs.add(master)

        self._poles_networks = networks

        return networks

    def get_total_operator_count(
        self,
        pole: str,
        poles_report: dict[str, PoleData]
    ) -> int:
        """
        Подсчитывает общее количество групп операторов с учетом, что
        опора может быть источником питания для другой опоы и надо проверить
        есть ли там еще доп. операторы.
        """
        poles_net = self._get_poles_networks().get(pole)

        if not poles_net:
            raise NotMasterPole(pole)

        return sum(
            poles_report[pl]['operator_group_count'] for pl in poles_net
        )
