import argparse
from typing import Optional


def str2bool(v: Optional[str]) -> bool:
    if v is None:
        return True
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f'Неверное значение: {v}. Ожидалось True/False.'
        )


def arguments_parser() -> tuple[bool]:
    parser = argparse.ArgumentParser(
        description='Дорасчёт интегральных показаний и профилей мощности.'
    )
    parser.add_argument(
        '--power-calc',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        dest='power_calc',
        help='Запустить расчет интегральных показаний',
    )
    parser.add_argument(
        '--power-profile-calc',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        dest='power_profile_calc',
        help='Запустить расчет профилей мощности'
    )

    args = parser.parse_args()

    return args.power_calc, args.power_profile_calc
