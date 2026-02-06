from typing import Optional
from .exceptions import ConfigEnvError

from .logger import app_logger


class Config:
    @staticmethod
    def validate_env_variables(env_vars: dict[str, Optional[str]]):
        """
        Проверка переменных окружения.
        Args:
            env_vars (dict): Словарь с переменными окружения.

        Example:
            ```python
            env_vars {
                'WEB_SECRET_KEY': 'my_secret_key',
                'WEB_BOT_EMAIL_LOGIN': 'bot@email.com',
                'WEB_BOT_EMAIL_PSWD': None,
                'EMAIL_SERVER': 'microsoft@outlook.com',
            }  # raise ConfigEnvError
            ```
        """
        missing_vars = [
            var_name
            for var_name, var_value in env_vars.items()
            if var_value is None
        ]

        if missing_vars:
            try:
                raise ConfigEnvError(missing_vars)
            except ConfigEnvError as e:
                app_logger.critical(e)
                raise
