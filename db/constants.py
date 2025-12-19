import os
from typing import Optional, TypedDict

from dotenv import load_dotenv

load_dotenv(override=True)

TS_DATABASE_URL = os.getenv('TS_DATABASE_URL')


class PoleData(TypedDict):
    id: int
    power_source_pole: Optional[str]
    is_master: bool
    is_standalone: bool
    operator_group_count: int
