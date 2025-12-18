from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .constants import TS_DATABASE_URL

ts_engine = create_engine(
    TS_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

TSSessionLocal = sessionmaker(
    bind=ts_engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()
