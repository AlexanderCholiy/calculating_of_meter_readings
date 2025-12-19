from calc.calc import MeterReadingsCalculator
from core.logger import app_logger
from core.wraps import timer
from db.utils import check_db_connection


@timer(app_logger)
def main():
    if not check_db_connection():
        return

    calculator = MeterReadingsCalculator()
    calculator.calculations()


if __name__ == '__main__':
    main()
