from sqlalchemy import Column, Integer, Text

from .connection import TSBase


class Pole(TSBase):
    __tablename__ = 'Таблица опор'

    id = Column('Ключ строки', Integer, primary_key=True)
    pole = Column('Шифр', Text, nullable=False)
    power_source_pole = Column('Источник питания', Text, nullable=True)


class Operator(TSBase):
    __tablename__ = 'EI.Размещённые арендаторы'

    id = Column('Ключ строки', Integer, primary_key=True)
    pole = Column('Шифр опоры', Text)
    base_station = Column('Имя БС/Оборудование', Text)
    operator = Column('Оператор', Text)
    operator_group = Column('Группа операторов', Text)
