from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from db.connection import ts_engine

from core.logger import db_logger


def check_db_connection() -> bool:
    try:
        with ts_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    except OperationalError:
        db_logger.error(
            'Ошибка соединения с базой данных TowerStore. '
            'Проверьте параметры подключения.'
        )
        return False

    except SQLAlchemyError:
        db_logger.exception('Ошибка соединения с базой данных TowerStore.')
        return False
