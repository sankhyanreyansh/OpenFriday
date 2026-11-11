"""
Glassmorphic Status Pill HUD for Open FRIDAY.
Provides a floating status indicator in the top-right corner of the desktop,
displaying real-time gesture states, modifier presence, and a master pause/resume toggle.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication

from src.control import permissions
from src.vision.gesture_recognizer import GestureState, GestureData

STATUS_PILL_STYLESHEET = """
QWidget#hudRoot {
    background: transparent;
}

QWidget#pillCard {
    background-color: rgba(18, 20, 26, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 19px;
}

QLabel#statusLabel {
    color: #E4E4E7;
    font-family: "Helvetica Neue", Helvetica, Arial;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.3px;
    background: transparent;
    border: none;
}

QPushButton#iconBtn {
    background-color: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 11px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    padding: 0px;
    font-size: 10px;
    color: #A1A1AA;
}

QPushButton#iconBtn:hover {
    background-color: rgba(255, 255, 255, 0.14);
    border-color: rgba(255, 255, 255, 0.20);
    color: #D4D4D8;
}

QPushButton#iconBtn:pressed {
    background-color: rgba(255, 255, 255, 0.04);
}
"""


class GlassmorphicStatusPill(QWidget):
    """Compact Glassmorphic Status Pill with real-time gesture feedback."""

    master_toggle_requested = pyqtSignal(bool)

    WINDOW_WIDTH = 220
    WINDOW_HEIGHT = 60

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open FRIDAY - Status Pill")
        self.setObjectName("hudRoot")
        self.setStyleSheet(STATUS_PILL_STYLESHEET)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.is_tracking_active = True
        self._last_display_text = ""
        self._last_color = ""
        self._build_ui()

        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self._reposition()

    def _reposition(self):
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        geo = screen.geometry()
        margin_top = 56
        margin_right = 16

        x = geo.x() + geo.width() - self.width() - margin_right
        y = geo.y() + margin_top
        self.move(int(x), int(y))

    def showEvent(self, event):
        super().showEvent(event)
        self._reposition()
        permissions.setup_macos_fullscreen_overlay(self)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Inner Pill Card (204 × 38 px with 19px border radius)
        self.pill_card = QWidget(self)
        self.pill_card.setObjectName("pillCard")
        self.pill_card.setFixedSize(204, 38)

        p_layout = QHBoxLayout(self.pill_card)
        p_layout.setContentsMargins(14, 0, 10, 0)
        p_layout.setSpacing(8)
        p_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.pill_status_lbl = QLabel("POINTING")
        self.pill_status_lbl.setObjectName("statusLabel")
        p_layout.addWidget(self.pill_status_lbl)

        p_layout.addStretch()

        self.mini_pause_btn = QPushButton("||")
        self.mini_pause_btn.setObjectName("iconBtn")
        self.mini_pause_btn.setToolTip("Pause / Resume Tracking")
        self.mini_pause_btn.clicked.connect(self.on_master_toggle)
        p_layout.addWidget(self.mini_pause_btn)

        root_layout.addWidget(self.pill_card)

    def on_master_toggle(self):
        self.is_tracking_active = not self.is_tracking_active
        if self.is_tracking_active:
            self.mini_pause_btn.setText("||")
            self._update_label("READY", "#E4E4E7")
        else:
            self.mini_pause_btn.setText(">")
            self._update_label("PAUSED", "#71717A")

        self.master_toggle_requested.emit(self.is_tracking_active)

    def on_status_changed(self, message: str):
        """Displays system status or error messages from the vision engine."""
        if not self.is_tracking_active:
            return
        msg_upper = message.upper()
        if "DICTATED" in msg_upper:
            color = "#34D399"
        elif "LISTENING" in msg_upper:
            color = "#F87171"
        elif "TRANSCRIBING" in msg_upper:
            color = "#60A5FA"
        elif "ERROR" in msg_upper or "FAILED" in msg_upper:
            color = "#F87171"
        else:
            color = "#A1A1AA"
        self._update_label(msg_upper, color)

    def _update_label(self, text: str, color: str):
        """Applies label text and styling with dirty checks to avoid redundant Qt re-styling."""
        if text != self._last_display_text:
            self.pill_status_lbl.setText(text)
            self._last_display_text = text

        if color != self._last_color:
            self.pill_status_lbl.setStyleSheet(f"color: {color}; font-weight: 500; font-size: 12px;")
            self._last_color = color

    def update_gesture_data(self, data: GestureData):
        """Updates live status indicator with current gesture state and modifier presence."""
        if not self.is_tracking_active:
            return

        state = data.state
        state_text = state.value

        if state == GestureState.DICTATION_LISTENING or data.is_dictating:
            display_text = "LISTENING..."
            color = "#F87171"
        elif state == GestureState.DICTATION_TRANSCRIBING:
            display_text = "TRANSCRIBING..."
            color = "#60A5FA"
        elif state == GestureState.RADIAL_MENU:
            if data.nav_action:
                display_text = data.nav_action
            elif data.radial_sector:
                display_text = data.radial_sector
            else:
                display_text = "RADIAL MENU"
            color = "#E4E4E7"
        elif state == GestureState.ZOOM:
            display_text = data.status_message.upper() if data.status_message else "ZOOM"
            color = "#E4E4E7"
        elif state == GestureState.SWIPE_NAV:
            if data.nav_action == "MISSION_CONTROL":
                display_text = "MISSION CONTROL"
            elif data.nav_action == "APP_EXPOSE":
                display_text = "APP EXPOSÉ"
            elif data.nav_action == "SPACE_LEFT":
                display_text = "SPACE LEFT"
            elif data.nav_action == "SPACE_RIGHT":
                display_text = "SPACE RIGHT"
            else:
                display_text = "3-FINGER SWIPE"
            color = "#E4E4E7"
        elif state == GestureState.NONE:
            if data.is_modifier_active:
                display_text = "[SHIFT] READY"
                color = "#D4D4D8"
            else:
                display_text = "SEARCHING..."
                color = "#71717A"
        elif data.is_modifier_active:
            display_text = f"[SHIFT] {state_text}"
            color = "#D4D4D8"
        else:
            display_text = state_text
            if state in (GestureState.CLICK, GestureState.DOUBLE_CLICK, GestureState.RIGHT_CLICK):
                color = "#E4E4E7"
            elif state in (GestureState.DRAGGING, GestureState.PINCHING, GestureState.SCROLLING):
                color = "#D4D4D8"
            else:
                color = "#A1A1AA"

        self._update_label(display_text, color)


# Backward-compatible alias
GlassmorphicHUDPanel = GlassmorphicStatusPill
