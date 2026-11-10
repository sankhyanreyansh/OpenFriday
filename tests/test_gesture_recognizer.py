"""
Comprehensive Unit and Regression Tests for GestureRecognizer.
Tests anatomical finger extension, pointing invariants, successive 3-finger swipes,
apex inflection detection, recoil suppression, and modifier exclusion.
"""

import pytest
from src.vision.gesture_recognizer import GestureRecognizer, GestureState


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def create_pointing_hand(mid_noise_factor=1.0, ring_noise_factor=1.0):
    """Generates 21 landmarks for a pointing hand (index extended, middle & ring curled)."""
    lms = [MockLandmark(0.5, 0.7, 0.0)] * 21
    lms[0] = MockLandmark(0.5, 0.7, 0.0)  # WRIST

    # Thumb
    lms[1] = MockLandmark(0.46, 0.65)
    lms[2] = MockLandmark(0.44, 0.60)
    lms[3] = MockLandmark(0.43, 0.56)
    lms[4] = MockLandmark(0.42, 0.52)   # THUMB_TIP

    # Index finger (extended UP)
    lms[5] = MockLandmark(0.48, 0.58)   # INDEX_MCP
    lms[6] = MockLandmark(0.48, 0.48)   # INDEX_PIP
    lms[7] = MockLandmark(0.48, 0.40)   # INDEX_DIP
    lms[8] = MockLandmark(0.48, 0.32)   # INDEX_TIP

    # Middle finger (curled)
    lms[9] = MockLandmark(0.52, 0.58)   # MIDDLE_MCP
    lms[10] = MockLandmark(0.52, 0.52)  # MIDDLE_PIP
    lms[11] = MockLandmark(0.52, 0.56)  # MIDDLE_DIP
    mid_y = 0.58 - 0.08 * (mid_noise_factor - 1.0)
    lms[12] = MockLandmark(0.52, mid_y)  # MIDDLE_TIP

    # Ring finger (curled)
    lms[13] = MockLandmark(0.55, 0.59)  # RING_MCP
    lms[14] = MockLandmark(0.55, 0.54)  # RING_PIP
    lms[15] = MockLandmark(0.55, 0.58)  # RING_DIP
    ring_y = 0.59 - 0.07 * (ring_noise_factor - 1.0)
    lms[16] = MockLandmark(0.55, ring_y) # RING_TIP

    # Pinky finger (curled)
    lms[17] = MockLandmark(0.58, 0.61)  # PINKY_MCP
    lms[18] = MockLandmark(0.58, 0.57)  # PINKY_PIP
    lms[19] = MockLandmark(0.58, 0.60)  # PINKY_DIP
    lms[20] = MockLandmark(0.58, 0.64)  # PINKY_TIP

    return lms


def create_three_finger_hand(mcp_x=0.5, mcp_y=0.5):
    """Generates 21 landmarks for 3 extended fingers (Index, Middle, Ring)."""
    lms = [MockLandmark(0.5, 0.7, 0.0)] * 21
    lms[0] = MockLandmark(mcp_x, mcp_y + 0.20, 0.0)  # WRIST

    # Thumb (open to side)
    lms[1] = MockLandmark(mcp_x - 0.06, mcp_y + 0.15)
    lms[2] = MockLandmark(mcp_x - 0.10, mcp_y + 0.10)
    lms[3] = MockLandmark(mcp_x - 0.13, mcp_y + 0.06)
    lms[4] = MockLandmark(mcp_x - 0.15, mcp_y + 0.03)  # THUMB_TIP

    # Index finger (extended)
    lms[5] = MockLandmark(mcp_x - 0.03, mcp_y)
    lms[6] = MockLandmark(mcp_x - 0.03, mcp_y - 0.08)
    lms[7] = MockLandmark(mcp_x - 0.03, mcp_y - 0.15)
    lms[8] = MockLandmark(mcp_x - 0.03, mcp_y - 0.22) # INDEX_TIP

    # Middle finger (extended)
    lms[9] = MockLandmark(mcp_x, mcp_y)
    lms[10] = MockLandmark(mcp_x, mcp_y - 0.09)
    lms[11] = MockLandmark(mcp_x, mcp_y - 0.16)
    lms[12] = MockLandmark(mcp_x, mcp_y - 0.24) # MIDDLE_TIP

    # Ring finger (extended)
    lms[13] = MockLandmark(mcp_x + 0.03, mcp_y)
    lms[14] = MockLandmark(mcp_x + 0.03, mcp_y - 0.08)
    lms[15] = MockLandmark(mcp_x + 0.03, mcp_y - 0.15)
    lms[16] = MockLandmark(mcp_x + 0.03, mcp_y - 0.21) # RING_TIP

    # Pinky finger (curled)
    lms[17] = MockLandmark(mcp_x + 0.06, mcp_y + 0.03)
    lms[18] = MockLandmark(mcp_x + 0.06, mcp_y - 0.01)
    lms[19] = MockLandmark(mcp_x + 0.06, mcp_y + 0.02)
    lms[20] = MockLandmark(mcp_x + 0.06, mcp_y + 0.05)

    return lms


def test_pointing_never_triggers_swipe():
    """Verify that pointing with index finger never triggers 3-finger swipe mode."""
    gr = GestureRecognizer()
    for i in range(25):
        lms = create_pointing_hand(mid_noise_factor=1.2, ring_noise_factor=1.15)
        res = gr.process(lms, (500.0, 400.0), now=1.0 + i * 0.016)
        assert res.state != GestureState.SWIPE_NAV


def test_three_finger_swipe_successive_left_flicks():
    """Verify 3 consecutive Left flicks trigger SPACE_RIGHT each time with recoil suppressed."""
    gr = GestureRecognizer()
    t = 10.0

    # 1. Enter 3-finger mode
    for i in range(3):
        t += 0.016
        lms = create_three_finger_hand(mcp_x=0.62, mcp_y=0.50)
        res = gr.process(lms, (500.0, 500.0), now=t)
    assert res.state == GestureState.SWIPE_NAV

    def simulate_flick_and_recoil(start_x, flick_target, recoil_target, current_t):
        local_t = current_t
        flick_action = None
        for step in range(5):
            local_t += 0.016
            x = start_x + (flick_target - start_x) * (step / 4.0)
            lms = create_three_finger_hand(mcp_x=x, mcp_y=0.50)
            r = gr.process(lms, (500.0, 500.0), now=local_t)
            if r.nav_action:
                flick_action = r.nav_action

        recoil_action = None
        for step in range(7):
            local_t += 0.016
            x = flick_target + (recoil_target - flick_target) * (step / 6.0)
            lms = create_three_finger_hand(mcp_x=x, mcp_y=0.50)
            r = gr.process(lms, (500.0, 500.0), now=local_t)
            if r.nav_action:
                recoil_action = r.nav_action

        return local_t, flick_action, recoil_action

    # Stroke 1
    t, a1, r1 = simulate_flick_and_recoil(0.62, 0.48, 0.62, t)
    assert a1 == "SPACE_RIGHT"
    assert r1 is None

    # Stroke 2 (immediate successive flick)
    t, a2, r2 = simulate_flick_and_recoil(0.62, 0.48, 0.62, t)
    assert a2 == "SPACE_RIGHT"
    assert r2 is None

    # Stroke 3 (third successive flick)
    t, a3, r3 = simulate_flick_and_recoil(0.62, 0.48, 0.62, t)
    assert a3 == "SPACE_RIGHT"
    assert r3 is None


def test_vertical_swipe_mission_control_and_recoil_guard():
    """Verify Up flick triggers MISSION_CONTROL and downward recoil is suppressed."""
    gr = GestureRecognizer()
    t = 20.0

    # Enter 3-finger mode
    for i in range(3):
        t += 0.016
        lms = create_three_finger_hand(mcp_x=0.50, mcp_y=0.55)
        gr.process(lms, (500.0, 500.0), now=t)

    # Flick UP (y decreases)
    mc_action = None
    for step in range(5):
        t += 0.016
        y = 0.55 - (0.12 * (step / 4.0))
        lms = create_three_finger_hand(mcp_x=0.50, mcp_y=y)
        res = gr.process(lms, (500.0, 500.0), now=t)
        if res.nav_action:
            mc_action = res.nav_action

    assert mc_action == "MISSION_CONTROL"

    # Recoil DOWN (y increases)
    recoil_action = None
    for step in range(7):
        t += 0.016
        y = 0.43 + (0.12 * (step / 6.0))
        lms = create_three_finger_hand(mcp_x=0.50, mcp_y=y)
        res = gr.process(lms, (500.0, 500.0), now=t)
        if res.nav_action:
            recoil_action = res.nav_action

    assert recoil_action is None


def test_left_hand_modifier_blocks_swipes():
    """Verify that raising the left hand strictly blocks 3-finger swipes."""
    gr = GestureRecognizer()
    lms_left = create_pointing_hand()
    lms_right = create_three_finger_hand(mcp_x=0.62, mcp_y=0.50)

    res = gr.process(lms_right, (500.0, 500.0), left_landmarks=lms_left, now=30.0)
    assert res.state != GestureState.SWIPE_NAV
    assert res.is_modifier_active is True



def create_pinched_hand():
    """Generates 21 landmarks for a pinched hand (thumb tip touching index tip)."""
    lms = create_pointing_hand()
    # Move thumb tip right next to index tip
    lms[4] = MockLandmark(lms[8].x + 0.01, lms[8].y + 0.01)
    return lms


def test_pinch_tap_triggers_click_not_drag():
    """Verify that a normal pinch tap with natural hand wobble triggers CLICK, never DRAGGING."""
    gr = GestureRecognizer()
    t = 40.0

    # 1. Pointing
    lms_open = create_pointing_hand()
    res = gr.process(lms_open, (500.0, 500.0), now=t)
    assert res.state == GestureState.POINTING

    # 2. Pinch starts
    lms_pinch = create_pinched_hand()
    t += 0.016
    res = gr.process(lms_pinch, (500.0, 500.0), now=t)
    assert res.state == GestureState.PINCHING

    # 3. Hold pinch for 220ms with natural tendon wobble of 18px
    for step in range(12):
        t += 0.016
        wobble_x = 500.0 + (step * 1.5)  # Max 18px wobble
        res = gr.process(lms_pinch, (wobble_x, 500.0), now=t)
        # CRUCIAL: Must NOT be confused for dragging!
        assert res.state == GestureState.PINCHING, f"Expected PINCHING, got {res.state}"

    # 4. Release pinch
    t += 0.016
    res = gr.process(lms_open, (518.0, 500.0), now=t)
    # Must trigger Left Click!
    assert res.state == GestureState.CLICK, f"Expected CLICK on tap release, got {res.state}"


def test_double_click_sequence():
    """Verify that two successive pinch taps trigger CLICK then DOUBLE_CLICK."""
    gr = GestureRecognizer()
    t = 50.0

    lms_open = create_pointing_hand()
    lms_pinch = create_pinched_hand()

    # Tap 1: Pinch and release
    gr.process(lms_open, (500.0, 500.0), now=t)
    t += 0.03
    gr.process(lms_pinch, (500.0, 500.0), now=t)
    t += 0.15
    res1 = gr.process(lms_open, (508.0, 502.0), now=t)
    assert res1.state == GestureState.CLICK, f"Tap 1 expected CLICK, got {res1.state}"

    # Gap between taps: 250ms
    for _ in range(15):
        t += 0.016
        gr.process(lms_open, (510.0, 502.0), now=t)

    # Tap 2: Pinch and release (Double click!)
    t += 0.03
    gr.process(lms_pinch, (512.0, 504.0), now=t)
    t += 0.15
    res2 = gr.process(lms_open, (514.0, 505.0), now=t)
    assert res2.state == GestureState.DOUBLE_CLICK, f"Tap 2 expected DOUBLE_CLICK, got {res2.state}"


def test_deliberate_drag_sequence():
    """Verify that holding pinch and moving >42px enters DRAGGING and exits cleanly without click."""
    gr = GestureRecognizer()
    t = 60.0

    lms_open = create_pointing_hand()
    lms_pinch = create_pinched_hand()

    # 1. Start pinch
    gr.process(lms_open, (200.0, 200.0), now=t)
    t += 0.02
    res = gr.process(lms_pinch, (200.0, 200.0), now=t)
    assert res.state == GestureState.PINCHING

    # 2. Move hand across screen: from (200, 200) to (300, 250) (>100px travel)
    drag_entered = False
    for step in range(20):
        t += 0.016
        x = 200.0 + (step * 5.0)
        y = 200.0 + (step * 2.5)
        res = gr.process(lms_pinch, (x, y), now=t)
        if res.state == GestureState.DRAGGING:
            drag_entered = True

    assert drag_entered is True, "Failed to enter DRAGGING state during deliberate hand movement"

    # 3. Release pinch after dragging
    t += 0.02
    res_rel = gr.process(lms_open, (300.0, 250.0), now=t)
    # Must NOT fire a click on drag release!
    assert res_rel.state != GestureState.CLICK, "Accidentally triggered click on drag release!"
    assert res_rel.state == GestureState.POINTING
    assert res_rel.status_message == "Drag Released"
