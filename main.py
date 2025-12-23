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
    except Exception as e:
        app_logger.exception(e)
        raise


if __name__ == '__main__':
    update_data = input(
        'Введите Y, чтобы запустить дорасчёт интегральных показаний: '
    ).strip().lower() if IS_EXE else 'y'

    if update_data != 'y':
        exit(1)

    main()

    while True and IS_EXE:
        exit_check = input('Введите Q, чтобы выйти из программы: ')
        if exit_check.strip().lower() == 'q':
            break
