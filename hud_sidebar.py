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
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (
    QFont,
    QColor,
    QGuiApplication,
    QFontMetrics,
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
        
        # Shift significantly further down and tighter to the right edge
        margin_top = 56     # Moved significantly further down
        margin_right = 4    # Moved significantly further right

        x = screen.x() + screen.width() - self.width() - margin_right
        y = screen.y() + margin_top

        self.move(int(x), int(y))

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

        self.is_flashing_transcribe = False
        self.ai_status: Optional[str] = None

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

    def flash_transcribing(self):
        """Briefly flashes TRANSCRIBING... status when voice dictation finishes."""
        self.is_flashing_transcribe = True
        self.pill_status_lbl.setText("● TRANSCRIBING...")
        self.pill_status_lbl.setStyleSheet("color: #D4D4D8; font-weight: 500; font-size: 12px;")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(1200, self._stop_flash_transcribing)

    def _stop_flash_transcribing(self):
        self.is_flashing_transcribe = False

    def update_ai_status(self, status: str):
        """Updates status pill when Gemini AI is thinking or speaking."""
        if status in ("THINKING", "SPEAKING"):
            self.is_flashing_transcribe = False
            self.ai_status = status
            if status == "THINKING":
                self.pill_status_lbl.setText("● THINKING...")
                self.pill_status_lbl.setStyleSheet("color: #D4D4D8; font-weight: 500; font-size: 12px;")
            elif status == "SPEAKING":
                self.pill_status_lbl.setText("● FRIDAY SPEAKING...")
                self.pill_status_lbl.setStyleSheet("color: #E4E4E7; font-weight: 500; font-size: 12px;")
        else:
            self.ai_status = None

    def update_gesture_data(self, data: GestureData):
        """Updates live status indicator with dual-hand modifier, AI assistant, and dictation state."""
        if not self.is_tracking_active:
            return

        if self.is_flashing_transcribe:
            return

        if self.ai_status == "THINKING":
            self.pill_status_lbl.setText("● THINKING...")
            self.pill_status_lbl.setStyleSheet("color: #D4D4D8; font-weight: 500; font-size: 12px;")
            return
        elif self.ai_status == "SPEAKING":
            self.pill_status_lbl.setText("● FRIDAY SPEAKING...")
            self.pill_status_lbl.setStyleSheet("color: #E4E4E7; font-weight: 500; font-size: 12px;")
            return

        state_text = data.state.value

        if data.state == GestureState.AI_LISTENING:
            color = "#E4E4E7"
            display_text = "● ASKING FRIDAY..."
        elif data.state == GestureState.LISTENING:
            color = "#E4E4E7"
            display_text = "● LISTENING..."
        elif data.state == GestureState.TRANSCRIBING:
            color = "#D4D4D8"
            display_text = "● TRANSCRIBING..."
        elif data.state == GestureState.RADIAL_MENU:
            color = "#E4E4E7"
            if data.nav_action:
                display_text = f"● {data.nav_action}"
            elif getattr(data, 'radial_sector', None):
                display_text = f"● {data.radial_sector}"
            else:
                display_text = "● RADIAL MENU"
        elif data.state == GestureState.SWIPE_NAV:
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

        if data.state in (GestureState.CLICK, GestureState.DOUBLE_CLICK, GestureState.RIGHT_CLICK, GestureState.SWIPE_NAV, GestureState.LISTENING, GestureState.AI_LISTENING, GestureState.RADIAL_MENU):
            color = "#E4E4E7"
        elif data.state in (GestureState.DRAGGING, GestureState.PINCHING, GestureState.TRANSCRIBING):
            color = "#D4D4D8"
        elif data.state == GestureState.SCROLLING:
            color = "#D4D4D8"
        else:
            color = "#A1A1AA"

        self.pill_status_lbl.setStyleSheet(f"color: {color}; font-weight: 500; font-size: 12px;")


CONTEXT_CARD_STYLESHEET = """
QWidget {
    font-family: "Helvetica Neue", Helvetica, Arial;
    color: #A1A1AA;
}

QWidget#contextRoot {
    background: transparent;
}

QWidget#contextInnerCard {
    background-color: rgba(18, 20, 26, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
}

QLabel#captionLabel {
    color: #71717A;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    background: transparent;
    border: none;
}

QLabel#replyLabel {
    color: #A1A1AA;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.35;
    background: transparent;
    border: none;
}

QPushButton#closeBtn {
    background: transparent;
    border: none;
    color: #71717A;
    font-size: 11px;
    font-weight: bold;
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
    padding: 0px;
}

QPushButton#closeBtn:hover {
    color: #D4D4D8;
}

QLabel#thumbnailLabel {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    background-color: rgba(0, 0, 0, 0.3);
}
"""


class HUDContextCard(QWidget):
    """Companion Glassmorphic Card displaying snippet thumbnail and latest AI response."""

    context_cleared = pyqtSignal()

    CARD_WIDTH = 220
    MIN_HEIGHT = 38

    def __init__(self, anchor_pill: Optional[QWidget] = None):
        super().__init__()
        self.setObjectName("contextRoot")
        self.setStyleSheet(CONTEXT_CARD_STYLESHEET)
        self.anchor_pill = anchor_pill

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.has_image = False
        self.has_reply = False

        # Inactivity auto-dismiss timer (10 seconds)
        self.dismiss_timer = QTimer(self)
        self.dismiss_timer.setSingleShot(True)
        self.dismiss_timer.setInterval(10000)
        self.dismiss_timer.timeout.connect(self._start_fadeout)

        # Smooth fadeout property animation (400ms)
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self.fade_anim.setDuration(400)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_anim.finished.connect(self._on_fade_finished)

        self._build_ui()
        width = self.anchor_pill.width() if self.anchor_pill else self.CARD_WIDTH
        self.setFixedWidth(width)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self.hide()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.inner_card = QWidget(self)
        self.inner_card.setObjectName("contextInnerCard")
        self.inner_card.setFixedWidth(204)
        self.inner_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

        self.card_layout = QVBoxLayout(self.inner_card)
        self.card_layout.setContentsMargins(12, 10, 12, 12)
        self.card_layout.setSpacing(6)

        # 1. Thumbnail preview label (hidden if no image)
        self.thumbnail_lbl = QLabel()
        self.thumbnail_lbl.setObjectName("thumbnailLabel")
        self.thumbnail_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_lbl.setFixedHeight(76)
        self.thumbnail_lbl.hide()
        self.card_layout.addWidget(self.thumbnail_lbl)

        # 2. AI response text label
        self.reply_lbl = QLabel()
        self.reply_lbl.setObjectName("replyLabel")
        self.reply_lbl.setWordWrap(True)
        self.reply_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.reply_lbl.hide()
        self.card_layout.addWidget(self.reply_lbl)

        root_layout.addWidget(self.inner_card)

    def _calculate_required_height(self) -> int:
        """Calculates exact required content height using QFontMetrics to prevent clipping."""
        card_margins = self.card_layout.contentsMargins()
        h_pad = card_margins.left() + card_margins.right()
        v_pad = card_margins.top() + card_margins.bottom()
        available_text_width = max(10, 204 - h_pad)  # 204px inner card width - 24px padding = 180px

        total_card_h = v_pad

        # Add thumbnail height if visible
        if self.has_image and self.thumbnail_lbl.isVisible():
            total_card_h += self.thumbnail_lbl.height() + self.card_layout.spacing()

        # Add wrapped text bounding height if text is visible
        if self.has_reply and self.reply_lbl.isVisible() and self.reply_lbl.text():
            metrics = QFontMetrics(self.reply_lbl.font())
            bounding_rect = metrics.boundingRect(
                0, 0, int(available_text_width), 10000,
                int(Qt.TextFlag.TextWordWrap),
                self.reply_lbl.text()
            )
            total_card_h += bounding_rect.height() + 6

        return int(max(total_card_h, self.MIN_HEIGHT))

    def update_layout_and_position(self, status_pill: Optional[QWidget] = None, calculated_height: Optional[int] = None, gap: int = 4):
        """Locks the top position strictly beneath the status pill using a single atomic setGeometry call."""
        pill = status_pill or self.anchor_pill
        if calculated_height is None:
            calculated_height = self._calculate_required_height()

        if pill:
            # Pill card (38px tall) is centered inside anchor_pill (60px tall) -> 11px margin top/bottom
            pill_visual_bottom = pill.y() + (pill.height() - 38) // 2 + 38
            target_x = pill.x()
            target_y = pill_visual_bottom + gap
            target_w = pill.width()
        else:
            screen = QGuiApplication.primaryScreen().geometry()
            margin_top = 56 + 11 + 38 + gap
            margin_right = 4
            target_w = self.CARD_WIDTH
            target_x = screen.x() + screen.width() - target_w - margin_right
            target_y = screen.y() + margin_top

        self.inner_card.setFixedHeight(int(calculated_height))
        # Apply position and size simultaneously to lock the top edge on macOS Cocoa
        self.setGeometry(int(target_x), int(target_y), int(target_w), int(calculated_height))
        self.updateGeometry()

    def reposition(self, status_pill: Optional[QWidget] = None):
        self.update_layout_and_position(status_pill)

    def _reposition(self):
        self.update_layout_and_position(self.anchor_pill)

    def _update_geometry(self):
        self.update_layout_and_position(self.anchor_pill)

    def _reset_fade_state(self):
        """Cancels running timers/animations and restores full opacity."""
        self.dismiss_timer.stop()
        self.fade_anim.stop()
        self.setWindowOpacity(1.0)

    def _start_fadeout(self):
        """Initiates smooth fadeout animation."""
        if not self.isVisible():
            return
        self.fade_anim.stop()
        self.fade_anim.setStartValue(self.windowOpacity())
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.start()

    def _on_fade_finished(self):
        """Called when fadeout animation completes."""
        if self.windowOpacity() <= 0.05:
            self.dismiss()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_layout_and_position(self.anchor_pill)
        permissions.setup_macos_fullscreen_overlay(self)

    def set_thumbnail(self, image_bytes: bytes):
        """Displays scaled thumbnail of captured snippet."""
        if not image_bytes:
            return
        self._reset_fade_state()
        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap()
        if pixmap.loadFromData(image_bytes):
            scaled_pixmap = pixmap.scaled(
                180, 76,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.thumbnail_lbl.setPixmap(scaled_pixmap)
            self.thumbnail_lbl.show()
            self.has_image = True
            self.update_layout_and_position(self.anchor_pill)
            self.show()
            self.raise_()

    def set_ai_reply(self, text: str):
        """Displays FRIDAY's latest spoken response in card and starts 10s auto-dismiss timer."""
        if not text or not text.strip():
            return
        self._reset_fade_state()
        clean_text = text.strip()
        self.reply_lbl.setText(clean_text)
        self.reply_lbl.show()
        if not self.has_image:
            self.thumbnail_lbl.clear()
            self.thumbnail_lbl.hide()
        self.has_reply = True
        self.update_layout_and_position(self.anchor_pill)
        self.show()
        self.raise_()
        # Start 10s auto-dismiss timer
        self.dismiss_timer.start(10000)

    def clear_thumbnail(self):
        """Clears thumbnail preview when context is consumed."""
        self.thumbnail_lbl.clear()
        self.thumbnail_lbl.hide()
        self.has_image = False
        self.update_layout_and_position(self.anchor_pill)

    def dismiss(self):
        """Dismisses the context card and clears current image."""
        self._reset_fade_state()
        self.thumbnail_lbl.clear()
        self.thumbnail_lbl.hide()
        self.reply_lbl.clear()
        self.reply_lbl.hide()
        self.has_image = False
        self.has_reply = False
        self.hide()
        self.context_cleared.emit()





# Backward-compatible alias
GlassmorphicHUDPanel = GlassmorphicStatusPill


