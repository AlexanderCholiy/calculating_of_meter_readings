import os
import sys
from logging import DEBUG as DEBUG_MODE
from logging import INFO

from dotenv import load_dotenv

load_dotenv(override=True)

IS_EXE = getattr(sys, 'frozen', False)

GLOBAL_TIMEOUT = int(os.getenv('GLOBAL_TIMEOUT', 15 * 60))

TRANSPOLATION_PROFILE_RESULT = os.getenv(
    'TRANSPOLATION_PROFILE_RESULT', 'True'
) == 'True'

SCALE_RESTORED_ONLY = os.getenv(
    'SCALE_RESTORED_ONLY', 'True'
) == 'True'

# Проверяем, запущен ли скрипт как exe:
FILE_DIR = os.path.dirname(sys.executable) if IS_EXE else (
    os.path.dirname(os.path.abspath(__file__))
)
BASE_DIR = os.path.normpath(os.path.join(FILE_DIR, '..'))

DEBUG = os.getenv('DEBUG', 'False') == 'True'

DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

TMP_DIR = os.path.join(DATA_DIR, 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

LOG_DIR = os.path.join(BASE_DIR, 'logs')

DEFAULT_LOG_MODE = 4 if DEBUG_MODE else 1
DEFAULT_LOG_LEVEL = DEBUG_MODE if DEBUG else INFO

DEFAULT_LOG_FILE = os.path.join(LOG_DIR, 'log.log')
DEFAULT_ROTATING_LOG_FILE = os.path.join(LOG_DIR, 'app', 'app.log')

os.makedirs(os.path.dirname(DEFAULT_LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DEFAULT_ROTATING_LOG_FILE), exist_ok=True)

DB_ROTATING_LOG_FILE = os.path.join(LOG_DIR, 'db', 'db.log')
os.makedirs(os.path.dirname(DB_ROTATING_LOG_FILE), exist_ok=True)

CALC_ROTATING_LOG_FILE = os.path.join(LOG_DIR, 'calc', 'calc.log')
os.makedirs(os.path.dirname(CALC_ROTATING_LOG_FILE), exist_ok=True)

EMAIL_ROTATING_LOG_FILE = os.path.join(LOG_DIR, 'emails', 'email.log')
os.makedirs(os.path.dirname(EMAIL_ROTATING_LOG_FILE), exist_ok=True)
