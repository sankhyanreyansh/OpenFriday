"""
Velocity-Adaptive EMA & One-Euro Motion Smoothing for FRIDAY.
Implements instantaneous hand velocity scaling with heavy stationary dampening (alpha ~ 0.03),
smooth non-abrupt velocity progression (up to 0.92), and an enforced 6px stationary deadzone.
"""

import math
import time
from typing import Tuple, Optional


class VelocityAdaptiveEMAFilter:
    """
    Velocity-Adaptive Exponential Moving Average (EMA) filter.
    Heavily dampens slow/micro movements while tracking fast motions with zero latency.
    """

    def __init__(
        self,
        base_alpha: float = 0.03,
        velocity_scale: float = 0.0018,
        deadzone_px: float = 6.0,
    ):
        self.base_alpha = float(base_alpha)
        self.velocity_scale = float(velocity_scale)
        self.deadzone_px = float(deadzone_px)

        self.prev_smooth_x: Optional[float] = None
        self.prev_smooth_y: Optional[float] = None
        self.prev_raw_x: Optional[float] = None
        self.prev_raw_y: Optional[float] = None
        self.prev_time: Optional[float] = None

    def update_params(self, min_cutoff: float, beta: float):
        """Updates base alpha and velocity responsiveness."""
        self.base_alpha = max(0.01, min(0.40, float(min_cutoff)))
        self.velocity_scale = max(0.0001, float(beta))

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        if timestamp is None:
            timestamp = time.time()

        if self.prev_smooth_x is None or self.prev_time is None:
            self.prev_smooth_x = x
            self.prev_smooth_y = y
            self.prev_raw_x = x
            self.prev_raw_y = y
            self.prev_time = timestamp
            return x, y

        dt = max(timestamp - self.prev_time, 0.001)
        self.prev_time = timestamp

        # If tracking was interrupted (gap > 150ms), snap to target
        if dt > 0.15:
            self.prev_smooth_x = x
            self.prev_smooth_y = y
            self.prev_raw_x = x
            self.prev_raw_y = y
            return x, y

        # 1. Calculate Instantaneous Velocity in screen pixels/sec:
        # v = sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2) / dt
        dx = x - self.prev_raw_x
        dy = y - self.prev_raw_y
        v = math.hypot(dx, dy) / dt
        self.prev_raw_x = x
        self.prev_raw_y = y

        # 2. Smooth progressive alpha scaling without abrupt jumps:
        # alpha = clamp(base_alpha + (v * velocity_scale), base_alpha, 0.92)
        alpha = max(self.base_alpha, min(0.92, self.base_alpha + (v * self.velocity_scale)))

        # 3. Apply smoothed coordinates:
        # pos_smooth = alpha * pos_raw + (1 - alpha) * pos_smooth_prev
        smooth_x = alpha * x + (1.0 - alpha) * self.prev_smooth_x
        smooth_y = alpha * y + (1.0 - alpha) * self.prev_smooth_y

        # 4. Enforced 6px Stationary Deadzone:
        # If delta < 6px, keep position strictly locked to previous frame
        delta = math.hypot(smooth_x - self.prev_smooth_x, smooth_y - self.prev_smooth_y)
        if delta < self.deadzone_px:
            return self.prev_smooth_x, self.prev_smooth_y

        # Huge jump snap protection (> 450px)
        if delta > 450.0:
            self.prev_smooth_x = x
            self.prev_smooth_y = y
            return x, y

        self.prev_smooth_x = smooth_x
        self.prev_smooth_y = smooth_y
        return smooth_x, smooth_y

    def reset(self):
        self.prev_smooth_x = None
        self.prev_smooth_y = None
        self.prev_raw_x = None
        self.prev_raw_y = None
        self.prev_time = None


Point2DOneEuroFilter = VelocityAdaptiveEMAFilter


class OneEuroFilter:
    """1D Adaptive low-pass filter."""

    def __init__(self, min_cutoff: float = 0.03, beta: float = 0.0018):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.prev_val: Optional[float] = None
        self.prev_raw: Optional[float] = None
        self.prev_time: Optional[float] = None

    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.time()
        if self.prev_val is None or self.prev_time is None:
            self.prev_val = value
            self.prev_raw = value
            self.prev_time = timestamp
            return value

        dt = max(timestamp - self.prev_time, 0.001)
        self.prev_time = timestamp

        v = abs(value - self.prev_raw) / dt
        self.prev_raw = value

        alpha = max(self.min_cutoff, min(0.92, self.min_cutoff + v * self.beta))
        res = alpha * value + (1.0 - alpha) * self.prev_val
        self.prev_val = res
        return res

    def reset(self):
        self.prev_val = None
        self.prev_raw = None
        self.prev_time = None
