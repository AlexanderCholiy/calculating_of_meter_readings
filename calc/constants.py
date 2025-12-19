import os
from datetime import datetime
from typing import Optional, TypedDict

from dotenv import load_dotenv

from core.constants import DATA_DIR

load_dotenv(override=True)

OUTPUT_CALC_FILE = os.path.join(DATA_DIR, 'calculations.xlsx')

AVG_EXP = float(os.getenv('AVG_EXP', 1200))

if AVG_EXP < 0:
    raise ValueError(
        'Константа AVG_EXP не может быть меньше нуля. '
        'Проверьте значение в .env файле.'
    )


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
