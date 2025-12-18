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
