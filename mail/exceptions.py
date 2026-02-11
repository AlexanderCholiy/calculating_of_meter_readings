class ValidationError(Exception):
    """Ошибка валидации"""


class EmptyEmailSelect(Exception):
    """
    Исключение возникает, когда при запросе к почтовому серверу
    по search-запросу не найдено ни одного письма.
    """

    def __init__(self, search: str):
        self.search = search
        super().__init__(
            f'Не найдено ни одного email, соответствующего запросу: {search}'
        )
