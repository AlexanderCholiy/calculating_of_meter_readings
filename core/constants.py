import os
from logging import DEBUG as DEBUG_MODE
from logging import INFO

from dotenv import load_dotenv

load_dotenv(override=True)

DEBUG = os.getenv('DEBUG', 'False') == 'True'

BASE_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..'
    )
)

DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

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
