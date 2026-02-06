import time
import json

from .power_calc import MeterReadingsCalculator
from .constants import POWER_CALC_RESULT_FILE, POWER_CALC_RESULT_TTL
from core.pretty_print import PrettyPrint
from core.logger import calc_logger

from mail.email_parser import email_parser, email_config


class PowerProfileCalc:

    def __init__(self):
        self._power_by_pole: dict[str, float] | None = None

    @property
    def power_by_pole(self):
        if POWER_CALC_RESULT_FILE.exists():
            file_age = (
                time.time()
                - POWER_CALC_RESULT_FILE.stat().st_mtime
            )

            if file_age < POWER_CALC_RESULT_TTL:
                try:
                    with open(
                        POWER_CALC_RESULT_FILE, 'r', encoding='utf-8'
                    ) as f:
                        self._power_by_pole = json.load(f)

                    file_age_msg = PrettyPrint.format_seconds_2_human_time(
                        file_age
                    )
                    calc_logger.debug(
                        'Загружены результаты расчета интегральных показаний '
                        f'из файлового кэша возраст: ({file_age_msg})'
                    )
                    return self._power_by_pole

                except Exception as e:
                    calc_logger.warning(
                        f'Ошибка чтения кэша: {e}. '
                        'Запускаем расчет интегральных показаний.'
                    )

            calculator = MeterReadingsCalculator()
            self._power_by_pole = calculator.calculations()

            return self._power_by_pole

    def calculations(self):
        email_parser.parser(
            subject=email_config['EMAIL_SUBJECT'],
            days_before=email_config['EMAIL_DAYS_BEFORE'],
            sender_email=email_config['EMAIL_SENDER'],
        )

        power_by_pole = self.power_by_pole
