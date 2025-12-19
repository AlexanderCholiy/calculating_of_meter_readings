from typing import Sequence


class MissingColumnsError(Exception):
    """Возникает, если в Excel-файле отсутствуют обязательные столбцы."""
    def __init__(self, missing_columns: Sequence[str], filename: str):
        self.missing_columns = missing_columns
        super().__init__(
            f'В файле {filename} отсутствуют обязательные столбцы: '
            f'{", ".join(missing_columns)}'
        )


class ExcelSaveError(PermissionError):
    """Исключение, возникающее при ошибках сохранения DataFrame в Excel."""
    def __init__(self, file_path: str, message: str = None):
        self.file_path = file_path
        default_message = (
            f'Не удалось сохранить результаты в файл {file_path}.'
        )
        super().__init__(message or default_message)
