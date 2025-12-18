from typing import TypedDict, Optional

from sqlalchemy.orm import Session, aliased
from sqlalchemy import case, exists, func
from db.models import Pole, Operator


class PoleData(TypedDict):
    id: int
    power_source_pole: Optional[str]
    is_master: bool
    operator_count: int


class PoleReport:

    @staticmethod
    def get_poles_with_master_flag(session: Session) -> dict[str, PoleData]:
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

        pole_qs = (
            session.query(
                Pole.id,
                Pole.pole,
                Pole.power_source_pole,
                is_master,
                func.count(Operator.id).label('operator_count'),
            )
            .outerjoin(Operator, Operator.pole == Pole.pole)
            .group_by(Pole)
        ).all()

        poles_report = {}

        for row in pole_qs:
            poles_report[row .pole] = {
                'id': row.id,
                'power_source_pole': row.power_source_pole,
                'is_master': row.is_master,
                'operator_count': row.operator_count,
            }

        return poles_report
