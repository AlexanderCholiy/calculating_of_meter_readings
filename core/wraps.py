import functools
import threading
import time
from datetime import datetime
from logging import Logger
from typing import Callable

from .utils import format_seconds
from .constants import IS_EXE


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


def retry(
    logger: Logger,
    retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 1.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Декоратор для повторного выполнения функции при ошибках.

    Args:
        logger (Logger): Логгер для записи ошибок.
        retries (int): Количество повторных попыток (по умолчанию 3)
        delay (float): Задержка между попытками в секундах (по умолчанию 1.0)
        exceptions: Кортеж исключений, при которых повторяем вызов
        передваваемой функции
        backoff_factor (float): Экспоненциальная задержка.
        sub_func_name str: Имя подфункции для логирования.

    Особенности:
        Если передать в качестве kwarg sub_func_name, это имя будет
        использовано для логирования метода класса.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            sub_func_name = kwargs.pop('sub_func_name', None)
            msg_sub_func_name = (
                f'(метод {sub_func_name}) '
            ) if sub_func_name else ''

            while True:
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except exceptions as e:
                    attempt += 1
                    if attempt > retries:
                        raise

                    current_delay = delay * (backoff_factor ** (attempt - 1))

                    msg = (
                        f'Ошибка {e.__class__.__name__}. '
                        f'Попытка {attempt}/{retries}, '
                        f'пробуем запустить {func.__name__} '
                        f'{msg_sub_func_name}'
                        # f'с параметрами args={args} kwargs={kwargs} '
                        f'снова через {format_seconds(current_delay)}'
                    )
                    logger.warn(msg, exc_info=False)
                    time.sleep(current_delay)
        return wrapper
    return decorator


def timeout(seconds: int):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = []
            exception = []

            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                raise TimeoutError(
                    f'Функция {func.__name__} превысила лимит {seconds} сек.'
                )

            if exception:
                raise exception[0]

            return result[0] if result else None
        return wrapper
    return decorator


def handle_exceptions(logger: Logger):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.exception(f'Критическая ошибка в {func.__name__}: {e}')

                if not IS_EXE:
                    raise
            return None
        return wrapper
    return decorator
