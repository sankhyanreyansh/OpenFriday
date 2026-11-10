"""
Tests for 1€ Dynamic Filter and CursorSmoother.
"""

import time
import pytest
from src.vision.one_euro_filter import (
    OneEuroFilter,
    CursorSmoother,
    smoothing_factor,
    exponential_smoothing,
)


def test_smoothing_factor():
    """Verify smoothing factor limits."""
    # At zero sample interval, alpha should approach 0
    alpha_0 = smoothing_factor(0.0, 1.0)
    assert alpha_0 == 0.0

    # Normal sample interval: 16ms, 1.0 Hz cutoff
    alpha_normal = smoothing_factor(0.016, 1.0)
    assert 0.0 < alpha_normal < 1.0


def test_exponential_smoothing():
    """Verify exponential smoothing weighted interpolation."""
    # Alpha = 1.0 -> current value
    assert exponential_smoothing(1.0, 10.0, 5.0) == 10.0
    # Alpha = 0.0 -> previous value
    assert exponential_smoothing(0.0, 10.0, 5.0) == 5.0
    # Alpha = 0.5 -> midpoint
    assert exponential_smoothing(0.5, 10.0, 5.0) == 7.5


def test_one_euro_filter_jitter_reduction():
    """Verify filter smooths small stationary signal jitter."""
    f = OneEuroFilter(t0=0.0, x0=100.0, min_cutoff=1.0, beta=0.01)

    # Simulate noisy stationary sensor (alternating +/- 0.5px)
    smoothed_vals = []
    for i in range(1, 30):
        t = i * 0.016
        jitter_x = 100.0 + (0.5 if i % 2 == 0 else -0.5)
        filtered = f.filter(t, jitter_x)
        smoothed_vals.append(filtered)

    # Variance of filtered values should be lower than input noise
    deviations = [abs(v - 100.0) for v in smoothed_vals[5:]]
    assert max(deviations) < 0.35


def test_one_euro_filter_dynamic_adaptation():
    """Verify filter dynamically increases cutoff on fast movements to eliminate lag."""
    f = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=0.05)

    # Fast step input: moves from 0 to 500 in 3 frames
    t = 0.016
    out1 = f.filter(t, 200.0)
    t += 0.016
    out2 = f.filter(t, 400.0)
    t += 0.016
    out3 = f.filter(t, 500.0)

    # Filtered output should aggressively track the step with minimal delay
    assert out3 > 420.0


def test_cursor_smoother_update_params_preserves_state():
    """Verify update_params does not destroy historical tracking state."""
    smoother = CursorSmoother(min_cutoff=1.0, beta=0.025)

    # Send initial coordinates
    sx, sy = smoother.smooth(500.0, 300.0)
    assert smoother.initialized is True

    # Move slightly
    time.sleep(0.01)
    sx2, sy2 = smoother.smooth(505.0, 302.0)
    assert smoother.initialized is True

    # Update sensitivity parameters live
    smoother.update_params(min_cutoff=1.5, beta=0.035, d_cutoff=2.0)

    # Crucial check: initialized must remain True, historical position must not reset to (0, 0)
    assert smoother.initialized is True
    assert smoother.min_cutoff == 1.5
    assert smoother.beta == 0.035
    assert smoother.d_cutoff == 2.0

    # Next smooth call should produce a value near 505, not near 0
    time.sleep(0.01)
    sx3, sy3 = smoother.smooth(506.0, 303.0)
    assert abs(sx3 - 506.0) < 5.0

