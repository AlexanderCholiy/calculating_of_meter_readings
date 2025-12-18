import functools
from datetime import datetime
from logging import Logger
from typing import Callable

from .utils import format_seconds


def timer(logger: Logger, is_debug: bool = True) -> Callable:
    """Декоратор для измерения и логирования времени выполнения функции."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                return func(*args, **kwargs)
            finally:
                execution_time = datetime.now() - start_time
                total_seconds = execution_time.total_seconds()
                msg = (
                    f'Время выполнения {func.__name__}: '
                    f'{format_seconds(total_seconds)}'
                )

                logger.debug(msg) if is_debug else logger.info(msg)

        return wrapper
    return decorator
