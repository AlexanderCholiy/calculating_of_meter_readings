from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from core.logger import db_logger
from db.connection import ts_engine


def check_db_connection():
    try:
        with ts_engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    except OperationalError:
        db_logger.error(
            'Ошибка соединения с базой данных TowerStore. '
            'Проверьте параметры подключения.'
        )
        raise

    except SQLAlchemyError:
        db_logger.exception('Ошибка соединения с базой данных TowerStore.')
        raise
