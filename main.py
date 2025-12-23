from calc.calc import MeterReadingsCalculator
from core.logger import app_logger
from core.wraps import timer
from db.utils import check_db_connection

from core.constants import IS_EXE


@timer(app_logger)
def main():
    if not check_db_connection():
        return

    calculator = MeterReadingsCalculator()
    calculator.calculations()


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
