from datetime import datetime, timedelta


def is_boundary_period(start_date: datetime, end_date: datetime) -> bool:
    """
    Проверяет, является ли период дат 'start_date' - 'end_date'
    строгим переходом от 1-го числа предыдущего месяца к 1-му числу текущего.
    """
    if start_date > end_date:
        return False

    first_day_current = end_date.replace(day=1)

    if end_date != first_day_current:
        return False

    last_day_prev_month = first_day_current - timedelta(days=1)

    expected_start_date = last_day_prev_month.replace(day=1)

    return start_date == expected_start_date
