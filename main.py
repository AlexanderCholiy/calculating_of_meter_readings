import sys

from calc.calc import MeterReadingsCalculator
from core.constants import IS_EXE
from core.logger import app_logger
from core.wraps import timer
from db.utils import check_db_connection


@timer(app_logger)
def main():
    try:
        check_db_connection()
        calculator = MeterReadingsCalculator()
        calculator.calculations()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        app_logger.exception(e)
        if not IS_EXE:
            raise


if __name__ == '__main__':
    update_data = input(
        'Нажмите Enter, чтобы запустить дорасчёт интегральных показаний: '
    ).strip().lower() if IS_EXE else ''

    if update_data:
        sys.exit(1)

    main()

    while True and IS_EXE:
        exit_check = (
            input('Нажмите Enter, чтобы выйти из программы: ').strip().lower()
        )
        if not exit_check:
            break
