import numpy as np


class ConfigRestoreSignal:

    def __init__(
        self,
        x: np.ndarray,
        noise_std_ratio: float,
        min_block_ratio: float,
        iterations: int = 5,
    ):
        pass
        # --- автоопределение шага ---
        self.sampling_seconds = int(np.median(np.diff(x)))

        # --- автоопределение периода ---
        self.total_duration = x[-1] - x[0]
        self.period_seconds = min(86400, self.total_duration)

        # --- точки в периоде ---
        self.points_per_period = self.period_seconds // self.sampling_seconds

        # --- автонастройки ---
        self.large_gap_seconds = 2 * self.period_seconds
        self.max_harmonics = min(6, self.points_per_period // 2)
        self.iterations = iterations

        # --- настройки шума ---
        self.noise_std_ratio = noise_std_ratio
        self.min_block_ratio = min_block_ratio
