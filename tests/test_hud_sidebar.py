"""
Unit tests for GlassmorphicStatusPill HUD label rendering and clean formatting.
"""

from PyQt6.QtWidgets import QApplication
import pytest

from src.ui.hud_sidebar import GlassmorphicStatusPill
from src.vision.gesture_recognizer import GestureData, GestureState


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_hud_status_pill_no_emojis(qapp):
    """Verify HUD pill status messages contain zero emojis across states."""
    pill = GlassmorphicStatusPill()

    # 1. Dictation listening
    data_listen = GestureData(state=GestureState.DICTATION_LISTENING, is_dictating=True)
    pill.update_gesture_data(data_listen)
    assert pill.pill_status_lbl.text() == "LISTENING..."

    # 2. Dictation transcribing
    data_trans = GestureData(state=GestureState.DICTATION_TRANSCRIBING)
    pill.update_gesture_data(data_trans)
    assert pill.pill_status_lbl.text() == "TRANSCRIBING..."

    # 3. Idle searching
    data_idle = GestureData(state=GestureState.NONE, is_modifier_active=False)
    pill.update_gesture_data(data_idle)
    assert pill.pill_status_lbl.text() == "SEARCHING..."

    # 4. Shift modifier ready
    data_shift = GestureData(state=GestureState.NONE, is_modifier_active=True)
    pill.update_gesture_data(data_shift)
    assert pill.pill_status_lbl.text() == "[SHIFT] READY"

    # 5. Dictated completion notification
    pill.on_status_changed("Dictated: hello world")
    assert pill.pill_status_lbl.text() == "DICTATED: HELLO WORLD"

