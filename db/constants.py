import os
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv

from core.constants import TMP_DIR

load_dotenv(override=True)

TS_DATABASE_URL = os.getenv('TS_DATABASE_URL')


class PoleData(TypedDict):
    id: int
    power_source_pole: Optional[str]
    is_master: bool
    is_standalone: bool
    operator_group_count: int


RAISE_TS_POLE_TABLE_LIMIT = 50_000

POLES_REPORT_CACHE_TTL = 3600

POLES_REPORT_CACHE_CACHE_FILE = Path(
    os.path.join(TMP_DIR, '__cache_poles_report.json')
)
