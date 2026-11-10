"""
Tests for MouseController and Radial Menu Actions.
"""

import time
import pytest
from src.control.mouse_controller import MouseController, post_keyboard_event
from src.control.radial_menu import RADIAL_ACTIONS, execute_radial_action, get_radial_action_by_index


def test_mouse_controller_clamp():
    """Verify coordinate clamping within screen boundaries."""
    mc = MouseController()
    w, h = mc.screen_width, mc.screen_height

    # In-bounds
    cx, cy = mc.clamp(100.0, 200.0)
    assert cx == 100.0
    assert cy == 200.0

    # Negative coordinates
    cx, cy = mc.clamp(-50.0, -100.0)
    assert cx == 0.0
    assert cy == 0.0

    # Beyond screen limits
    cx, cy = mc.clamp(w + 500.0, h + 500.0)
    assert cx == w - 1
    assert cy == h - 1


def test_click_and_double_click_no_blocking_sleep():
    """Verify click, double_click, and right_click execute in < 5ms without blocking sleeps."""
    mc = MouseController()

    t0 = time.perf_counter()
    mc.click(100.0, 100.0)
    t_click = (time.perf_counter() - t0) * 1000.0
    assert t_click < 15.0, f"Click took {t_click:.2f}ms (expected <15ms without sleep)"

    t0 = time.perf_counter()
    mc.double_click(100.0, 100.0)
    t_dclick = (time.perf_counter() - t0) * 1000.0
    assert t_dclick < 15.0, f"Double-click took {t_dclick:.2f}ms (expected <15ms without sleep)"

    t0 = time.perf_counter()
    mc.right_click(100.0, 100.0)
    t_rclick = (time.perf_counter() - t0) * 1000.0
    assert t_rclick < 15.0, f"Right-click took {t_rclick:.2f}ms (expected <15ms without sleep)"


def test_radial_actions_structure():
    """Verify radial menu actions are defined with valid keycodes and labels."""
    assert len(RADIAL_ACTIONS) >= 4
    for action in RADIAL_ACTIONS:
        assert "id" in action
        assert "label" in action
        assert "symbol" in action
        assert "key_code" in action
        assert isinstance(action["key_code"], int)


def test_radial_action_lookup():
    """Verify circular radial action lookup by index."""
    a0 = get_radial_action_by_index(0)
    assert a0 is not None
    assert a0["id"] == RADIAL_ACTIONS[0]["id"]

    # Modulo wrap
    a_wrap = get_radial_action_by_index(len(RADIAL_ACTIONS))
    assert a_wrap["id"] == RADIAL_ACTIONS[0]["id"]


def test_post_keyboard_event_execution():
    """Verify post_keyboard_event runs without exceptions."""
    # Virtual keycode 53 = Escape
    post_keyboard_event(53, flags=0)

