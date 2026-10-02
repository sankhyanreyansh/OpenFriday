"""
One-Euro Dynamic Smoothing & Adaptive Motion Filtering for FRIDAY.
Implements the 1€ Filter algorithm for human-computer interaction,
eliminating jitter at low velocities while preserving zero-lag precision during rapid movements.
"""

import math
import time
from typing import Tuple, Optional


def smoothing_factor(t_e: float, cutoff: float) -> float:
    """Calculates alpha smoothing coefficient from sample interval and cutoff frequency."""
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    """Computes low-pass exponential smoothing."""
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter:
    """
    1D One-Euro Filter.
    Adapts cutoff frequency dynamically based on signal derivative (speed).
    """

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def filter(self, t: float, x: float) -> float:
        t_e = max(t - self.t_prev, 1e-5)
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class CursorSmoother:
    """
    2D Cursor Motion Smoother utilizing dual One-Euro filters for X and Y coordinates.
    Guarantees jitter-free stationary pointing and zero-lag tracking during high-speed gestures.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ):
        t = time.time()
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.filter_x = OneEuroFilter(t, 0.0, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
        self.filter_y = OneEuroFilter(t, 0.0, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
        self.initialized = False

    def smooth(self, raw_screen_x: float, raw_screen_y: float) -> Tuple[int, int]:
        """Filters 2D screen coordinates and returns integer pixel positions."""
        t = time.time()
        if not self.initialized:
            self.filter_x = OneEuroFilter(t, raw_screen_x, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            self.filter_y = OneEuroFilter(t, raw_screen_y, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            self.initialized = True
            return int(round(raw_screen_x)), int(round(raw_screen_y))

        smooth_x = self.filter_x.filter(t, raw_screen_x)
        smooth_y = self.filter_y.filter(t, raw_screen_y)
        return int(round(smooth_x)), int(round(smooth_y))

    def reset(self):
        """Resets smoother state when hand tracking is lost or re-acquired."""
        self.initialized = False

    def update_params(self, min_cutoff: float, beta: float, d_cutoff: Optional[float] = None):
        """Updates filter sensitivity parameters."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        if d_cutoff is not None:
            self.d_cutoff = float(d_cutoff)
        self.initialized = False


# Backwards compatibility
VelocityAdaptiveEMAFilter = CursorSmoother
Point2DOneEuroFilter = CursorSmoother
