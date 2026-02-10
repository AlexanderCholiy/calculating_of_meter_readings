from pathlib import Path
import shutil

import pandas as pd
from pandas import DataFrame

from .logger import app_logger, calc_logger
from calc.exceptions import ExcelSaveError


def format_seconds(seconds: float) -> str:
    """Форматирует время в секундах в читаемый вид."""
    if seconds < 0.001:
        return f'{seconds * 1000:.0f} мс'
    elif seconds < 1:
        return f'{seconds * 1000:.1f} мс'
    elif seconds < 60:
        if seconds < 10:
            return f'{seconds:.2f} сек'
        else:
            return f'{seconds:.1f} сек'
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f'{minutes} мин {remaining_seconds:.1f} сек'
    elif seconds < 86400:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f'{hours} ч {remaining_minutes} мин {remaining_seconds:.0f} сек'
    else:
        days = int(seconds // 86400)
        remaining_hours = int((seconds % 86400) // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f'{days} дн {remaining_hours} ч {remaining_minutes} мин'


def clear_folder(folder_path: str | Path):
    folder = Path(folder_path)

    if not folder.exists():
        return

    for item in folder.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            app_logger.debug(f'Не удалось удалить {item}: {e}')


def get_sorted_excel_files(
    directory: Path, reverse: bool = True
) -> list[Path]:
    """
    Находит все Excel-файлы в директории и сортирует их по имени.
    По умолчанию самые свежие файлы (с датой в начале имени) будут первыми.
    """
    files = [
        f for f in directory.iterdir()
        if f.is_file() 
        and f.suffix.lower() in ('.xls', '.xlsx')
        and not f.name.startswith('~$')  # исключаем временные файлы Excel ~$
    ]

    # Поскольку формат даты %Y-%m-%d в начале имени,
    # обычная сортировка строк совпадает с хронологической.
    files.sort(key=lambda x: x.name, reverse=reverse)

    return files


def write_to_excel(
    file_path: Path,
    sheet_name: str,
    df: DataFrame,
    **to_excel_kwargs
):
    """
    Универсальный внутренний метод для записи листа в Excel.
    Автоматически определяет: создать новый файл или обновить лист в 
    существующем.
    """
    try:
        file_exists = file_path.exists()
        mode = 'a' if file_exists else 'w'

        writer_args = {'engine': 'openpyxl', 'mode': mode}

        sheet_exists = False
        if file_exists:
            writer_args['if_sheet_exists'] = 'replace'
            with pd.ExcelFile(file_path, engine='openpyxl') as reader:
                sheet_exists = sheet_name in reader.sheet_names

        with pd.ExcelWriter(file_path, **writer_args) as writer:
            df.to_excel(writer, sheet_name=sheet_name, **to_excel_kwargs)

        action = 'обновлен' if sheet_exists else 'добавлен'
        if not file_exists:
            action = 'создан'

        calc_logger.info(
            f'Лист "{sheet_name}" успешно {action} в файле {file_path.name}'
        )

    except PermissionError:
        raise ExcelSaveError(
            file_path=file_path,
            message=f'Файл {file_path.name} открыт в другой программе.'
        )
