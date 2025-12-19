from core.logger import app_logger
from core.wraps import timer
from db.connection import TSSessionLocal
from db.reports.poles_report import PoleReport
from db.utils import check_db_connection
from calc.calc import MeterReadingsCalculator


@timer(app_logger)
def main():
    # if not check_db_connection():
    #     return

    # with TSSessionLocal() as session:
    #     poles_report = PoleReport.get_poles_with_master_flag(session)

    # print(poles_report.get('10981-5-77-1-275'))
    calculator = MeterReadingsCalculator()
    calculator.calculations()


if __name__ == '__main__':
    main()
