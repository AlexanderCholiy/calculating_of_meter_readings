from typing import Optional

import random


class ProfileAlgoritm:

    def full_empty_algoritm(
        self,
        vars: list[Optional[float]],
        good_profiles: dict[str, list[float]],
        good_profile_keys: list[str],
    ) -> Optional[list[float]]:
        """Полностью отсутсвуют данные за период"""
        if not set(vars).issubset({None, 0, 0.0}):
            return

        use_by_readings = random.choice(good_profile_keys)
        random_sample = good_profiles[use_by_readings]

        print(random_sample)
