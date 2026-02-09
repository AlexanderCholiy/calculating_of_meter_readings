import os
from pathlib import Path

from core.constants import DATA_DIR


MAX_ATTACHMENT_SIZE = 50 * 1024 * 1024  # 50 MB

ALLOWED_MIME_PREFIXES = {'application/vnd.ms-excel'}

ALLOWED_EXTENSIONS = {'.xls', '.xlsx'}

EMAIL_DIR = Path(os.path.join(DATA_DIR, 'emails'))
os.makedirs(EMAIL_DIR, exist_ok=True)

FILENAME_DATETIME_PREFIX = '%Y-%m-%d_%H-%M'
