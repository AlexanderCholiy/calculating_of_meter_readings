import random
from datetime import datetime

import numpy as np
from scipy.interpolate import PchipInterpolator


class ProfileAlgoritm:

    def __init__(self, scale_restored_only_val: bool, **kwargs):
        super().__init__(**kwargs)
        self.scale_restored_only_val = scale_restored_only_val

    def _finalize_processing(
        self, y: np.ndarray, y_interp: np.ndarray, total_power: float
    ) -> np.ndarray:
        y_interp = np.clip(y_interp, 0, None)

        y_scaled = (
            self.scale_restored_only(y, y_interp, total_power)
            if self.scale_restored_only_val
            else self.scale_to_area(y_interp, total_power)
        )

        return np.clip(y_scaled, 0, None)

    def full_empty_algoritm(
        self,
        good_profiles: dict[float, np.ndarray],
        good_profile_keys: list[float],
        total_power: float,
    ) -> np.ndarray:
        """
        Полностью отсутствуют данные за период.
        Генерирует профиль на основе одного из эталонных профилей,
        масштабируя его так, чтобы сумма соответствовала total_power.
        """
        sample_total_power = random.choice(good_profile_keys)
        random_sample = good_profiles[sample_total_power]

        # масштабируем профиль по времени и площади
        try:
            scaled_profile = self.scale_to_area(random_sample, total_power)
        except ValueError:
            # если площадь равна 0, возвращаем нули
            scaled_profile = np.zeros(len(random_sample), dtype=float)

        return scaled_profile

    def interpolate_inside_algoritm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        total_power: float,
    ) -> np.ndarray:
        y_interp = self.interpolate_inside(x, y)
        return self._finalize_processing(y, y_interp, total_power)

    def stretch_algoritm(
        self,
        y: np.ndarray,
        total_power: float,
    ) -> np.ndarray:
        y_interp = self.stretch_known_fill_nans(y)
        return self._finalize_processing(y, y_interp, total_power)

    def mixed_fill_algoritm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        total_power: float,
    ) -> np.ndarray:
        y_interp = self.mixed_fill(x, y)
        return self._finalize_processing(y, y_interp, total_power)

    @staticmethod
    def datetime_to_seconds(x: list[datetime]) -> np.ndarray:
        t0 = x[0]
        return np.array([(dt - t0).total_seconds() for dt in x])

    @staticmethod
    def interpolate_inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(y)
        valid_count = mask.sum()

        if valid_count == 0:
            raise ValueError('Нет опорных точек для интерполяции')

        if valid_count == 1:
            return np.full(len(y), y[mask][0], dtype=float)

        if valid_count == 2:
            return np.interp(x, x[mask], y[mask])

        # защита от нуля и отрицательных значений
        eps = 1e-9
        y_safe = np.maximum(y[mask], eps)

        # логарифмируем
        log_y = np.log(y_safe)

        # строим монотонный сплайн в лог-пространстве
        spline = PchipInterpolator(x[mask], log_y, extrapolate=False)

        # интерполируем
        log_y_interp = spline(x)

        # возвращаемся из лог-пространства
        y_interp = np.exp(log_y_interp)

        # заменяем только NaN
        y_filled = y.copy()
        y_filled[np.isnan(y_filled)] = y_interp[np.isnan(y_filled)]

        return y_filled

    @staticmethod
    def stretch_inside_fill_nans(y: np.ndarray) -> np.ndarray:
        """
        Заполняет только ВНУТРЕННИЕ NaN методом равномерного растяжения
        известных значений внутри диапазона между первой и последней
        известной точкой.

        Краевые NaN не изменяются.
        Исходные известные значения сохраняются.
        """
        known_mask = ~np.isnan(y)

        if not known_mask.any():
            raise ValueError("Нет известных точек для заполнения")

        known_indices = np.where(known_mask)[0]
        first_idx = known_indices[0]
        last_idx = known_indices[-1]

        # Если нет внутренней области — ничего не делаем
        if last_idx - first_idx < 2:
            return y.copy()

        y_result = y.copy()

        # Внутренний сегмент
        inner_slice = slice(first_idx, last_idx + 1)
        y_inner = y[inner_slice]

        inner_known_mask = ~np.isnan(y_inner)
        inner_nan_mask = np.isnan(y_inner)

        known_y = y_inner[inner_known_mask]

        # Растягиваем только внутри сегмента
        stretched_inner = np.interp(
            np.arange(len(y_inner)),
            np.linspace(0, len(y_inner) - 1, num=len(known_y)),
            known_y
        )

        # Заполняем только внутренние NaN
        y_inner[inner_nan_mask] = stretched_inner[inner_nan_mask]

        y_result[inner_slice] = y_inner

        return y_result

    @staticmethod
    def stretch_known_fill_nans(y: np.ndarray) -> np.ndarray:
        """
        Заполняет пропущенные значения (NaN) в массиве y методом растяжения
        известных значений.
        """
        y = np.array(y, dtype=float)

        # Индексы известных и NaN точек
        known_mask = ~np.isnan(y)
        nan_mask = np.isnan(y)

        # Берём только известные значения
        known_y = y[known_mask]

        # Растягиваем их на весь профиль равномерно
        stretched_y = np.interp(
            np.arange(len(y)),
            np.linspace(0, len(y) - 1, num=len(known_y)),
            known_y
        )

        # Вставляем значения только в те места, где был NaN
        y[nan_mask] = stretched_y[nan_mask]

        return y

    @staticmethod
    def mixed_fill(x: np.ndarray, y: np.ndarray):
        """
        Комбинированное заполнение пропусков (NaN):

        - Внутренние NaN (между первой и последней известной точкой)
        заполняются методом interpolate_inside (PCHIP).
        - Краевые NaN (слева и справа от известной области)
        заполняются методом stretch_known_fill_nans.

        Исходные известные значения не изменяются.
        """
        y = np.array(y, dtype=float)
        mask = ~np.isnan(y)

        valid_count = mask.sum()
        if valid_count == 0:
            raise ValueError('Нет известных точек для заполнения')

        if valid_count == 1:
            return np.full(len(y), y[mask][0], dtype=float)

        first_idx = np.where(mask)[0][0]
        last_idx = np.where(mask)[0][-1]

        y_result = y.copy()

        # ---- 1. Внутренняя интерполяция ----
        if last_idx - first_idx > 1:
            # Берём только внутренний диапазон
            x_inner = x[first_idx:last_idx + 1]
            y_inner = y[first_idx:last_idx + 1]

            y_inner_filled = ProfileAlgoritm.interpolate_inside(
                x_inner, y_inner
            )
            y_result[first_idx:last_idx + 1] = y_inner_filled

        # ---- 2. Краевая "растяжка" ----
        if first_idx > 0 or last_idx < len(y) - 1:
            y_stretched = ProfileAlgoritm.stretch_known_fill_nans(y)
            edge_mask = np.isnan(y)
            y_result[edge_mask] = y_stretched[edge_mask]

        return y_result

    @staticmethod
    def scale_to_area(
        y: np.ndarray, target_area: float
    ) -> np.ndarray:
        current_area = y.sum()

        if current_area == 0:
            raise ValueError('Текущая площадь равна 0')

        factor = target_area / current_area

        return y * factor

    @staticmethod
    def scale_restored_only(
        y_original: np.ndarray,
        y_filled: np.ndarray,
        target_area: float,
    ) -> np.ndarray:
        """
        Масштабирует ТОЛЬКО восстановленные точки (которые были NaN в
        оригинале), не изменяя исходные известные значения.

        НЕ гарантирует, что итоговая сумма будет равна target_area.
        """
        mask_known = ~np.isnan(y_original)
        mask_restored = np.isnan(y_original)

        sum_known = y_original[mask_known].sum()
        sum_restored = y_filled[mask_restored].sum()

        if sum_restored == 0:
            raise ValueError('Сумма восстановленных значений равна 0')

        target_restored_sum = target_area - sum_known

        factor = target_restored_sum / sum_restored

        y_result = y_filled.copy()
        y_result[mask_restored] *= factor

        return y_result
