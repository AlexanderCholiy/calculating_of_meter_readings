import sys

from calc.power_calc import MeterReadingsCalculator
from calc.power_profile_calc import PowerProfileCalc
from core.args_parser import arguments_parser
from core.constants import (
    GLOBAL_TIMEOUT,
    IS_EXE,
    SCALE_RESTORED_ONLY,
    TRANSPOLATION_PROFILE_RESULT
)
from core.logger import app_logger
from core.wraps import handle_exceptions, timeout, timer


@timeout(GLOBAL_TIMEOUT)
@timer(app_logger)
@handle_exceptions(app_logger)
def run_power_calc():
    calculator = MeterReadingsCalculator()
    calculator.calculations()


@timeout(GLOBAL_TIMEOUT)
@timer(app_logger)
@handle_exceptions(app_logger)
def run_power_profile_calc():
    calculator = PowerProfileCalc(SCALE_RESTORED_ONLY)
    calculator.calculations(TRANSPOLATION_PROFILE_RESULT)


if __name__ == '__main__':
    power_calc, power_profile_calc = arguments_parser()

    if not any((power_calc, power_profile_calc)):
        sys.exit(1)

    if IS_EXE and power_calc and not power_profile_calc:
        update_data = input(
            'Нажмите Enter, чтобы запустить дорасчёт интегральных показаний: '
        ).strip().lower() if IS_EXE else ''
        if update_data:
            sys.exit(1)

    elif IS_EXE and not power_calc and power_profile_calc:
        update_data = input(
            'Нажмите Enter, чтобы запустить дорасчёт профилей мощности: '
        ).strip().lower() if IS_EXE else ''
        if update_data:
            sys.exit(1)

    elif IS_EXE and power_calc and power_profile_calc:
        update_data = input(
            'Нажмите Enter, чтобы запустить дорасчёт интегральных показаний и '
            'профилей мощности:'
        ).strip().lower() if IS_EXE else ''
        if update_data:
            sys.exit(1)

    if power_calc:
        try:
            run_power_calc()
        except TimeoutError as e:
            app_logger.critical(e)

    if power_profile_calc:
        try:
            run_power_profile_calc()
        except TimeoutError as e:
            app_logger.critical(e)

    while True and IS_EXE:
        exit_check = (
            input('Нажмите Enter, чтобы выйти из программы: ').strip().lower()
        )
        if not exit_check:
            break
