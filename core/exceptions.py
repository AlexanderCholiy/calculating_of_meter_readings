class LoggerError(Exception):
    """Ошибка выбора режима работы логгера."""


class ConfigEnvError(Exception):
    """Исключение для отсутствующих переменных конфигурации."""

    def __init__(self, missing_vars: list[str]):
        self.missing_vars = missing_vars
        missing_vars_str = ', '.join(missing_vars)
        super().__init__(
            'Ошибка конфигурации. Отсутствуют переменные '
            f'{missing_vars_str} в .env файле.'
        )
