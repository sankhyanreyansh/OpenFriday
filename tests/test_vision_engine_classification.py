"""
Unit tests for VisionEngine hand classification and dual-hand disambiguation.
Verifies dominant right hand vs modifier left hand assignment, persistence across frames,
and fallback logic.
"""

from typing import List
from src.vision.vision_engine import VisionEngine
from src.control.mouse_controller import MouseController


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class MockCategory:
    def __init__(self, label: str, score: float = 0.99):
        self.label = label
        self.category_name = label
        self.score = score


class MockHandedness:
    def __init__(self, label: str):
        self.classification = [MockCategory(label)]


def create_hand(base_x: float, base_y: float) -> List[MockLandmark]:
    """Generates 21 mock landmarks with palm centered around base_x, base_y."""
    lms = [MockLandmark(base_x, base_y)] * 21
    lms[0] = MockLandmark(base_x, base_y + 0.1)  # WRIST
    lms[5] = MockLandmark(base_x - 0.04, base_y)  # INDEX_MCP
    lms[9] = MockLandmark(base_x, base_y)        # MIDDLE_MCP
    lms[13] = MockLandmark(base_x + 0.04, base_y) # RING_MCP
    lms[17] = MockLandmark(base_x + 0.08, base_y + 0.02) # PINKY_MCP
    return lms


def test_classify_single_dominant_hand():
    ctrl = MouseController()
    engine = VisionEngine(camera_id=0, mouse_controller=ctrl)
    engine.mirror_horizontal = True  # Default camera mirror mode

    hand_right = create_hand(0.7, 0.5)
    # When mirror_horizontal=True, target_right_label is "Left"
    handedness = [MockHandedness("Left")]

    right, left = engine._classify_hands([hand_right], handedness)
    assert right is not None, "Single dominant hand should be returned as right hand"
    assert left is None, "Left modifier hand should be None"


def test_classify_single_modifier_hand():
    ctrl = MouseController()
    engine = VisionEngine(camera_id=0, mouse_controller=ctrl)
    engine.mirror_horizontal = True

    hand_left = create_hand(0.2, 0.5)
    # When mirror_horizontal=True, target_left_label is "Right"
    handedness = [MockHandedness("Right")]

    right, left = engine._classify_hands([hand_left], handedness)
    assert right is None, "Right hand should be None when only modifier hand is visible"
    assert left is not None, "Single modifier hand should be returned as left hand"


def test_classify_dual_hands_and_centroid_persistence():
    ctrl = MouseController()
    engine = VisionEngine(camera_id=0, mouse_controller=ctrl)
    engine.mirror_horizontal = True

    hand_dom = create_hand(0.7, 0.5)
    hand_mod = create_hand(0.2, 0.5)

    handedness = [MockHandedness("Left"), MockHandedness("Right")]

    # Frame 1: Initial classification
    right, left = engine._classify_hands([hand_dom, hand_mod], handedness)
    assert right is hand_dom
    assert left is hand_mod
    assert engine.prev_dominant_centroid is not None
    assert engine.prev_modifier_centroid is not None

    # Frame 2: Even if handedness info is missing/swapped, centroid tracking maintains identity
    moved_dom = create_hand(0.72, 0.51)
    moved_mod = create_hand(0.21, 0.49)
    # Swapped array order in detection
    right2, left2 = engine._classify_hands([moved_mod, moved_dom], [])
    assert right2 is moved_dom, "Centroid tracking must keep dominant hand identity"
    assert left2 is moved_mod, "Centroid tracking must keep modifier hand identity"


def test_single_right_hand_classified_correctly_to_stop_dictation():
    """Verify that raising a single right hand is classified as dominant hand so dictation stops."""
    ctrl = MouseController()
    engine = VisionEngine(camera_id=0, mouse_controller=ctrl)
    engine.mirror_horizontal = True

    # User was dictating with left hand
    engine.gesture_recognizer.is_dictating = True

    # User drops left hand and raises right hand (dominant label "Left" in mirrored mode)
    hand_right = create_hand(0.7, 0.5)
    right, left = engine._classify_hands([hand_right], [MockHandedness("Left")])
    assert right is hand_right, "Single right hand must be recognized as dominant hand"
    assert left is None, "Modifier hand must be None"

    # Verify that passing this right hand to gesture_recognizer immediately triggers STOP
    gesture_data = engine.gesture_recognizer.process(
        right, (500.0, 500.0), left_landmarks=None, now=10.0
    )
    assert gesture_data.dictation_event == "STOP", "Raising right hand must stop dictation"
    assert engine.gesture_recognizer.is_dictating is False

