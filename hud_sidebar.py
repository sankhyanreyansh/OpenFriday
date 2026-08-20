"""
Minimalist Monochrome Glassmorphic Status Pill for FRIDAY.
2-Layer Centered Architecture with 11px vertical and 8px horizontal breathing headroom.
Equal 20px top and right screen margins, eliminating all bottom and edge clipping.
"""

import sys
import os
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QApplication,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QFont,
    QColor,
    QGuiApplication,
)

from gesture_recognizer import GestureState, GestureData
import permissions

STATUS_PILL_STYLESHEET = """
QWidget {
    font-family: "Helvetica Neue", Helvetica, Arial;
    color: #D4D4D8;
}

QWidget#hudRoot {
    background: transparent;
}

QWidget#pillCard {
    background-color: rgba(18, 20, 26, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 19px;
}

QLabel {
    color: #D4D4D8;
    font-size: 12px;
    font-weight: 400;
    background: transparent;
    border: none;
}

QLabel#statusLabel {
    color: #D4D4D8;
    font-weight: 500;
    font-size: 12px;
    letter-spacing: 0.3px;
}

QPushButton#iconBtn {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    font-size: 11px;
    padding: 0px;
    min-width: 24px;
    max-width: 24px;
    min-height: 24px;
    max-height: 24px;
    color: #A1A1AA;
}

QPushButton#iconBtn:hover {
    background-color: rgba(255, 255, 255, 0.12);
    border-color: rgba(255, 255, 255, 0.18);
    color: #D4D4D8;
}

QPushButton#iconBtn:pressed {
    background-color: rgba(255, 255, 255, 0.04);
}
"""


class GlassmorphicStatusPill(QWidget):
    """Compact Glassmorphic Status Pill with centered 2-layer architecture and equal 20px margins."""

    master_toggle_requested = pyqtSignal(bool)

    WINDOW_WIDTH = 220
    WINDOW_HEIGHT = 60
    MARGIN = 20

    def __init__(self):
        super().__init__()
        self.setObjectName("hudRoot")
        self.setStyleSheet(STATUS_PILL_STYLESHEET)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.is_tracking_active = True
        self._build_ui()

        # Set fixed size and position with equal 20px screen margins
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self._reposition()

    def _reposition(self):
        screen = QGuiApplication.primaryScreen().geometry()
        x = screen.x() + screen.width() - self.WINDOW_WIDTH - self.MARGIN
        y = screen.y() + self.MARGIN
        self.move(x, y)

    def showEvent(self, event):
        super().showEvent(event)
        self._reposition()
        permissions.setup_macos_fullscreen_overlay(self)

    def _build_ui(self):
        # 1. Root container with zero margins, perfectly centering the pill card
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 2. Inner Pill Card (Fixed 204 × 38 px with 19px border radius)
        self.pill_card = QWidget(self)
        self.pill_card.setObjectName("pillCard")
        self.pill_card.setFixedSize(204, 38)
        
        p_layout = QHBoxLayout(self.pill_card)
        p_layout.setContentsMargins(14, 0, 10, 0)
        p_layout.setSpacing(8)
        p_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.pill_status_lbl = QLabel("● POINTING")
        self.pill_status_lbl.setObjectName("statusLabel")
        p_layout.addWidget(self.pill_status_lbl)

        p_layout.addStretch()

        self.mini_pause_btn = QPushButton("⏸")
        self.mini_pause_btn.setObjectName("iconBtn")
        self.mini_pause_btn.setToolTip("Pause / Resume Tracking")
        self.mini_pause_btn.clicked.connect(self.on_master_toggle)
        p_layout.addWidget(self.mini_pause_btn)

        root_layout.addWidget(self.pill_card)

    def on_master_toggle(self):
        self.is_tracking_active = not self.is_tracking_active
        if self.is_tracking_active:
            self.mini_pause_btn.setText("⏸")
        else:
            self.mini_pause_btn.setText("▶")
            self.pill_status_lbl.setText("● PAUSED")
            self.pill_status_lbl.setStyleSheet("color: #71717A; font-weight: 500; font-size: 12px;")

        self.master_toggle_requested.emit(self.is_tracking_active)

    def update_gesture_data(self, data: GestureData):
        """Updates live status indicator with dual-hand modifier state."""
        if not self.is_tracking_active:
            return

        state_text = data.state.value

        if data.state == GestureState.SWIPE_NAV:
            color = "#E4E4E7"
            if data.nav_action == "MISSION_CONTROL":
                display_text = "● MISSION CONTROL"
            elif data.nav_action in ("SPACE_LEFT", "SPACE_RIGHT"):
                display_text = "● SPACES"
            else:
                display_text = "● SWIPE NAV"
        elif data.is_modifier_active:
            display_text = f"● [SHIFT] {state_text}"
        else:
            display_text = f"● {state_text}"

        self.pill_status_lbl.setText(display_text)

        if data.state in (GestureState.CLICK, GestureState.DOUBLE_CLICK, GestureState.RIGHT_CLICK, GestureState.SWIPE_NAV):
            color = "#E4E4E7"
        elif data.state in (GestureState.DRAGGING, GestureState.PINCHING):
            color = "#D4D4D8"
        elif data.state == GestureState.SCROLLING:
            color = "#D4D4D8"
        else:
            color = "#A1A1AA"

        self.pill_status_lbl.setStyleSheet(f"color: {color}; font-weight: 500; font-size: 12px;")


# Backward-compatible alias
GlassmorphicHUDPanel = GlassmorphicStatusPill
