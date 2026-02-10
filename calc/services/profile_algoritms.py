import random
import numpy as np
from datetime import datetime, timedelta

from typing import Optional
from scipy.interpolate import CubicSpline, PchipInterpolator
from math import sin, pi


class ProfileAlgoritm:

    def full_empty_algoritm(
        self,
        good_profiles: dict[float, list[float]],
        good_profile_keys: list[float],
        total_power: float,
    ) -> list[float]:
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

        return scaled_profile.tolist()

    def interpolate_inside_algoritm(
        self,
        x_dt: list[datetime],
        y: list[Optional[float]],
        total_power: float,
    ) -> list[float]:
        y_interp = np.asarray(
            self.interpolate_inside(x_dt, y),
            dtype=float
        )
        y_interp = self.scale_to_area(y_interp, total_power)

        y_interp = np.clip(y_interp, 0, None)

        return y_interp.tolist()

    def extrapolate_edges_algoritm(
        self,
        x_dt: list[datetime],
        y: list[Optional[float]],
        total_power: float,
    ) -> list[float]:
        y_interp = np.asarray(
            self.extrapolate_edges(x_dt, y),
            dtype=float
        )

        y_interp = np.clip(y_interp, 0, None)

        y_interp = self.scale_to_area(y_interp, total_power)

        return y_interp.tolist()

    def mixed_fill_algoritm(
        self,
        x_dt: list[datetime],
        y: list[Optional[float]],
        total_power: float,
    ) -> list[float]:
        y_interp = np.asarray(
            self.mixed_fill(x_dt, y),
            dtype=float,
        )
        y_interp = self.scale_to_area(y_interp, total_power)

        y_interp = np.clip(y_interp, 0, None)

        return y_interp.tolist()

    @staticmethod
    def datetime_to_seconds(x: list[datetime]) -> np.ndarray:
        t0 = x[0]
        return np.array([(dt - t0).total_seconds() for dt in x])

    @staticmethod
    def interpolate_inside(
        x_dt: list[datetime], y: list[Optional[float]]
    ) -> list[float]:
        x = ProfileAlgoritm.datetime_to_seconds(x_dt)
        y = np.asarray(y, dtype=float)

        mask = ~np.isnan(y)
        valid_count = mask.sum()

        if valid_count == 0:
            raise ValueError('Нет опорных точек для интерполяции')

        if valid_count == 1:
            return np.full(len(y), y[mask][0]).tolist()

        if valid_count == 2:
            y_interp = np.interp(x, x[mask], y[mask])
            return y_interp.tolist()

        spline = CubicSpline(
            x[mask],
            y[mask],
            extrapolate=False
        )

        y_interp = spline(x)

        # страховка — если вдруг появились NaN внутри диапазона
        nan_mask = np.isnan(y_interp)
        if nan_mask.any():
            y_interp[nan_mask] = np.interp(
                x[nan_mask],
                x[~nan_mask],
                y_interp[~nan_mask],
            )

        return y_interp.tolist()

    @staticmethod
    def extrapolate_edges(
        x_dt: list[datetime],
        y: list[Optional[float]],
        edge_points: int = 3,
        max_slope: float = 0.1,
    ):
        """
        Экстраполяция профиля мощности без NaN с автоматическим определением
        предыдущего периода (от первой до последней известной точки)
        """

        x = ProfileAlgoritm.datetime_to_seconds(x_dt)
        y = np.array([v if v is not None else np.nan for v in y], dtype=float)

        mask = ~np.isnan(y)
        if mask.sum() < 2:
            raise ValueError("Недостаточно опорных точек")

        xk = x[mask]
        yk = y[mask]

        # 1. PCHIP внутри известных точек
        pchip = PchipInterpolator(xk, yk, extrapolate=False)
        result = pchip(x)

        mean_profile = np.nanmean(yk)

        # 2. Левый край
        left_mask = x < xk[0]
        if left_mask.any():
            x_left = xk[:edge_points]
            y_left = yk[:edge_points]
            slope_left = np.polyfit(x_left, y_left, 1)[0]
            slope_left = np.clip(slope_left, -max_slope, max_slope)
            result[left_mask] = yk[0] + slope_left * (x[left_mask] - xk[0])
            decay_left = np.exp(-np.linspace(0, 3, left_mask.sum()))
            result[left_mask] = mean_profile + decay_left * (result[left_mask] - mean_profile)

        # 3. Правый край
        right_mask = x > xk[-1]
        if right_mask.any():
            x_right = xk[-edge_points:]
            y_right = yk[-edge_points:]
            slope_right = np.polyfit(x_right, y_right, 1)[0]
            slope_right = np.clip(slope_right, -max_slope, max_slope)
            result[right_mask] = yk[-1] + slope_right * (x[right_mask] - xk[-1])
            decay_right = np.exp(-np.linspace(0, 3, right_mask.sum()))
            result[right_mask] = mean_profile + decay_right * (result[right_mask] - mean_profile)

        # 4. Автоматический предыдущий период
        try:
            # период = длина между первой и последней известной точкой
            period_seconds = xk[-1] - xk[0]
            prev_dt = [dt - timedelta(seconds=period_seconds) for dt in x_dt]
            prev_seconds = ProfileAlgoritm.datetime_to_seconds(prev_dt)

            prev_left_mask = prev_seconds < xk[0]
            prev_right_mask = prev_seconds > xk[-1]

            if prev_left_mask.any():
                result[left_mask] = 0.5 * (result[left_mask] + y[prev_left_mask])
            if prev_right_mask.any():
                result[right_mask] = 0.5 * (result[right_mask] + y[prev_right_mask])
        except Exception:
            pass

        # 5. Любые оставшиеся NaN заменяем на среднее известных точек
        nan_mask = np.isnan(result)
        if nan_mask.any():
            result[nan_mask] = mean_profile

        return result

    @staticmethod
    def mixed_fill(x_dt: list[datetime], y: list[Optional[float]]):
        x = ProfileAlgoritm.datetime_to_seconds(x_dt)
        y = np.asarray(y, dtype=float)

        mask = ~np.isnan(y)
        valid_count = mask.sum()

        if valid_count == 0:
            raise ValueError('Все значения отсутствуют')

        if valid_count == 1:
            return np.full(len(y), y[mask][0])

        x_known = x[mask]
        y_known = y[mask]

        if valid_count == 2:
            return np.interp(
                x,
                x_known,
                y_known,
                left=None,
                right=None
            )

        spline = CubicSpline(
            x_known,
            y_known,
            extrapolate=True
        )

        y_filled = spline(x)

        nan_mask = np.isnan(y_filled)
        if nan_mask.any():
            y_filled[nan_mask] = np.interp(
                x[nan_mask],
                x[~nan_mask],
                y_filled[~nan_mask],
            )

        return y_filled

    @staticmethod
    def scale_to_area(
        y: list[Optional[float]], target_area: float
    ) -> np.ndarray:
        y = np.asarray(y, dtype=float)

        current_area = y.sum()

        if current_area == 0:
            raise ValueError('Текущая площадь равна 0')

        factor = target_area / current_area

        return y * factor
