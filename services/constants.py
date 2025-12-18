import os

from dotenv import load_dotenv

load_dotenv(override=True)

TS_DATABASE_URL = os.getenv('TS_DATABASE_URL')
