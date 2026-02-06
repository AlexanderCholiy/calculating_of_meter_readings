from sqlalchemy import case, exists, func
from sqlalchemy.orm import Session, aliased

from core.logger import db_logger
from core.pretty_print import PrettyPrint
from core.wraps import timer
from db.constants import (
    RAISE_TS_POLE_TABLE_LIMIT,
    PoleData,
)
from db.exceprions import EmptyTableError
from db.models import Operator, Pole


class PoleReport:

    @staticmethod
    @timer(db_logger, is_debug=False)
    def get_poles_with_master_flag(session: Session) -> dict[str, PoleData]:
        """
        Формирует отчет по опорам.

        Опора считается мастер-опорой (`is_master=True`), если:
        - она указана как источник питания (`power_source_pole`)
          хотя бы у одной другой опоры;
        - при этом у самой опоры источник питания не задан.

        Автономная опора (`is_standalone=True`):
        Опора считается автономной, если:
        - у неё не указан источник питания (`power_source_pole IS NULL`);
        - она нигде не используется как источник питания для других опор.

        Количество операторов (`operator_group_count`) определяется как число
        уникальных групп операторов, связанных с опорой, при этом:
        - не учитываются пустые значения;
        - не учитывается группа операторов "Other".

        Возвращает словарь, где ключ — шифр опоры, а значение — агрегированные
        данные по опоре.
        """
        timeout_limit_msg = PrettyPrint.format_seconds_2_human_time(5 * 60)
        PrettyPrint.warning_print(
            ('Запрашиваем данные в TowerStore.', False),
            ('Операция может занять продолжительное время', False),
            (f'({timeout_limit_msg}).', True),
        )

        PoleAlias = aliased(Pole)

        is_master = case(
            (
                exists().where(
                    (PoleAlias.power_source_pole == Pole.pole)
                    & (PoleAlias.power_source_pole.isnot(None))
                ),
                True
            ),
            else_=False
        ).label('is_master')

        is_standalone = case(
            (
                (~exists().where(PoleAlias.power_source_pole == Pole.pole))
                & (Pole.power_source_pole.is_(None)),
                True,
            ),
            else_=False,
        ).label('is_standalone')

        operator_group_count = func.count(
            func.distinct(
                case(
                    (
                        (Operator.operator_group.isnot(None))
                        & (Operator.operator_group != 'Other'),
                        Operator.operator_group,
                    ),
                    else_=None,
                )
            )
        ).label('operator_group_count')

        pole_qs = (
            session.query(
                Pole.id,
                Pole.pole,
                Pole.power_source_pole,
                is_master,
                is_standalone,
                operator_group_count,
            )
            .outerjoin(Operator, Operator.pole == Pole.pole)
            .group_by(Pole)
        ).all()

        total = len(pole_qs)
        if total < RAISE_TS_POLE_TABLE_LIMIT:
            raise EmptyTableError(Pole.__tablename__, total)

        poles_report = {}

        for index, row in enumerate(pole_qs):
            PrettyPrint.progress_bar_warning(
                index, total, 'Обработка данных из TowerStore:'
            )

            poles_report[row .pole] = {
                'id': row.id,
                'power_source_pole': row.power_source_pole,
                'is_master': row.is_master,
                'is_standalone': row.is_standalone,
                'operator_group_count': row.operator_group_count,
            }

        return poles_report
