from dataclasses import dataclass

import numpy as np


@dataclass
class DriveCycle:
    name: str
    t_s: np.ndarray
    speed_mps: np.ndarray
    grade_rad: np.ndarray


def _make_time_array(duration_s: float, dt_s: float) -> np.ndarray:
    return np.arange(0.0, duration_s + dt_s, dt_s)


def urban_cycle(dt_s: float = 1.0, duration_s: float = 1800.0) -> DriveCycle:
    t = _make_time_array(duration_s, dt_s)
    speed = np.zeros_like(t)

    for i, ti in enumerate(t):
        phase = ti % 300.0
        if phase < 30.0:
            speed[i] = (phase / 30.0) * 12.0
        elif phase < 90.0:
            speed[i] = 12.0
        elif phase < 120.0:
            speed[i] = 12.0 * (1.0 - (phase - 90.0) / 30.0)
        elif phase < 180.0:
            speed[i] = 0.0
        elif phase < 220.0:
            speed[i] = ((phase - 180.0) / 40.0) * 8.0
        elif phase < 270.0:
            speed[i] = 8.0
        else:
            speed[i] = 8.0 * (1.0 - (phase - 270.0) / 30.0)

    grade = 0.01 * np.sin(2.0 * np.pi * t / 900.0)
    return DriveCycle(name="urban", t_s=t, speed_mps=speed, grade_rad=np.arctan(grade))


def highway_cycle(dt_s: float = 1.0, duration_s: float = 2400.0) -> DriveCycle:
    t = _make_time_array(duration_s, dt_s)
    speed = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < 120.0:
            speed[i] = (ti / 120.0) * 28.0
        elif ti < 2100.0:
            speed[i] = 28.0 + 2.0 * np.sin(2.0 * np.pi * ti / 180.0)
        else:
            speed[i] = max(0.0, 28.0 * (1.0 - (ti - 2100.0) / 300.0))

    grade = 0.015 * np.sin(2.0 * np.pi * t / 1200.0)
    return DriveCycle(name="highway", t_s=t, speed_mps=speed, grade_rad=np.arctan(grade))


def mixed_cycle(dt_s: float = 1.0, duration_s: float = 3600.0) -> DriveCycle:
    t = _make_time_array(duration_s, dt_s)
    split = int(0.45 * t.size)

    urban = urban_cycle(dt_s=dt_s, duration_s=max(dt_s, split * dt_s - dt_s))
    hw = highway_cycle(dt_s=dt_s, duration_s=max(dt_s, (t.size - split) * dt_s - dt_s))

    speed = np.concatenate([urban.speed_mps, hw.speed_mps])
    grade = np.concatenate([urban.grade_rad, hw.grade_rad])

    speed = speed[: t.size]
    grade = grade[: t.size]

    return DriveCycle(name="mixed", t_s=t, speed_mps=speed, grade_rad=grade)


def get_cycle(name: str, dt_s: float, duration_s: float) -> DriveCycle:
    key = name.lower().strip()
    if key == "urban":
        return urban_cycle(dt_s=dt_s, duration_s=duration_s)
    if key == "highway":
        return highway_cycle(dt_s=dt_s, duration_s=duration_s)
    if key == "mixed":
        return mixed_cycle(dt_s=dt_s, duration_s=duration_s)
    raise ValueError(f"Unsupported cycle '{name}'. Use urban, highway, or mixed.")
