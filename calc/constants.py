import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv

from core.constants import DATA_DIR, TMP_DIR

load_dotenv(override=True)

OUTPUT_POWER_CALC_FILE = Path(
    os.path.join(DATA_DIR, 'power_calc.xlsx')
)

OUTPUT_POWER_PROFILE_CALC_FILE = Path(
    os.path.join(DATA_DIR, 'power_profile_calc.xlsx')
)

POWER_CALC_RESULT_FILE = Path(
    os.path.join(TMP_DIR, '__power_calc_result.json')
)

POWER_CALC_RESULT_TTL = 10 * 60

AVG_EXP = float(os.getenv('AVG_EXP', 1200))

if AVG_EXP < 0:
    raise ValueError(
        'Константа AVG_EXP не может быть меньше нуля. '
        'Проверьте значение в .env файле.'
    )

ROUND_CALCULATION_DIGITS = Decimal('0.00')


class PeriodReadingData(TypedDict):
    tu_type: str
    last_read: Optional[float]
    exp: Optional[float]
    start_date: datetime | None
    end_date: datetime | None


class IntegralReadingFile:
    INTEGRAL_READINGS_FILE = os.path.join(
        DATA_DIR, 'интегральные_показания.xlsx'
    )

    CODE_EO_COL_IN_INTEGRAL_READINGS = 'Код ЭО'
    PU_NUMBER_COL_IN_INTEGRAL_READINGS = 'Номер ПУ'
    PREV_READ_COL_IN_INTEGRAL_READINGS = 'Предыдущие показания'
    CURRENT_READ_COL_IN_INTEGRAL_READINGS = 'Текущие показания'

    REQUIRED_INTEGRAL_READINGS_COLUMNS = [
        CODE_EO_COL_IN_INTEGRAL_READINGS,
        PU_NUMBER_COL_IN_INTEGRAL_READINGS,
        PREV_READ_COL_IN_INTEGRAL_READINGS,
        CURRENT_READ_COL_IN_INTEGRAL_READINGS,
    ]


class ArchiveFile:
    ARCHIVE_FILE = os.path.join(DATA_DIR, 'архив.xlsx')

    CODE_EO_COL_IN_ARCHIVE = 'Код ЭО'
    POLE_COL_IN_ARCHIVE = 'шифр'
    ASKUE_COL_IN_ARCHIVE = 'АСКУЭ'

    REQUIRED_ARCHIVE_COLUMNS = [
        CODE_EO_COL_IN_ARCHIVE,
        POLE_COL_IN_ARCHIVE,
        ASKUE_COL_IN_ARCHIVE,
    ]

    EXTRA_COLUMNS = [
        s.strip() for s in os.getenv('ARCHIVE_EXTRA_COLUMNS', '').split(',')
        if s.strip()
    ]

    if EXTRA_COLUMNS:
        REQUIRED_ARCHIVE_COLUMNS.extend(EXTRA_COLUMNS)


class PeriodReadingFile:
    PERIOD_READINGS_FILE = os.path.join(DATA_DIR, 'показания_за_период.xlsx')

    POLE_COL_IN_PERIOD_READINGS = 'Наименование точки'
    PU_NUMBER_COL_IN_PERIOD_READINGS = 'Заводской номер прибора учета'
    TU_TYPE_COL_IN_PERIOD_READINGS = 'Тип точки учёта'
    LAST_READ_COL_IN_PERIOD_READINGS = 'Показания на конец периода'
    START_DATE_COL_IN_PERIOD_READINGS = 'Действительно на1'
    END_DATE_COL_IN_PERIOD_READINGS = 'Действительно на2'
    EXP_COL_IN_PERIOD_READINGS = 'Расход'

    REQUIRED_PERIOD_READINGS_COLUMNS = [
        POLE_COL_IN_PERIOD_READINGS,
        PU_NUMBER_COL_IN_PERIOD_READINGS,
        TU_TYPE_COL_IN_PERIOD_READINGS,
        LAST_READ_COL_IN_PERIOD_READINGS,
        START_DATE_COL_IN_PERIOD_READINGS,
        END_DATE_COL_IN_PERIOD_READINGS,
        EXP_COL_IN_PERIOD_READINGS,
    ]

    TU_TYPE_PRIORITY: dict[str, int] = {
        'коммерческий': 2,  # Чем больше, тем выше приоритет
        'контрольная': 1,
    }


class PowerProfileFile:
    POLE_COL_IN_POWER_PROFILE = 'Наименование объекта'
    BS_COL_IN_POWER_PROFILE = 'Базовая станция'
    PU_NUMBER_COL_IN_POWER_PROFILE = 'Прибор учета'

    USE_BY_METER_COL_IN_POWER_PROFILE = 'Потребление за период по показаниям'
    USE_BY_PROFILE_COL_IN_POWER_PROFILE = 'Потребление за период по профилю'

    DIF_COL_IN_POWER_PROFILE = 'Разница'

    DATETIME_FORMAT_POWER_PROFILE = '%d.%m.%Y %H:%M'

    PROFILE_SUMPLES_NUMBERS = 100
    MIN_KNOWN_POINTS_FRACTION = 0.1  # % от всех точек
