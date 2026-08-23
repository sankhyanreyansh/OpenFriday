"""
AR Visual Screen Annotation Canvas for Open FRIDAY.
Renders real-time glowing bounding boxes and floating glassmorphic callout cards
anchored directly adjacent to screen elements, diagrams, circuits, and UI features.
Includes a 10-second auto-dismiss timer and smooth opacity fadeout animations.
"""

from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QFontMetrics,
    QPainterPath,
    QGuiApplication,
)
import permissions


class AnnotationOverlay(QWidget):
    """Fullscreen transparent, click-through AR canvas for visual annotations."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(None)  # Top-level window
        self.setWindowTitle("Open FRIDAY - AR Annotation Canvas")

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._sync_geometry()
        self.active_annotations: List[Dict[str, Any]] = []

        # 10-Second Auto-Dismiss Timer
        self.fade_timer = QTimer(self)
        self.fade_timer.setSingleShot(True)
        self.fade_timer.timeout.connect(self.start_fadeout)
        self.dismiss_timer = self.fade_timer  # alias

        # Opacity Fadeout Animation
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.anim.finished.connect(self._on_fade_finished)
        self.fade_anim = self.anim  # alias

    def _sync_geometry(self):
        screen = QGuiApplication.primaryScreen()
        if screen:
            self.screen_rect = screen.geometry()
            self.setGeometry(self.screen_rect)
            self.screen_w = self.screen_rect.width()
            self.screen_h = self.screen_rect.height()
        else:
            self.screen_w = 1470
            self.screen_h = 956
            self.resize(self.screen_w, self.screen_h)

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_geometry()
        permissions.setup_macos_fullscreen_overlay(self)

    def display_annotations(self, annotations: List[Dict[str, Any]], spoken_response: str = ""):
        """Slot connected to signals.annotations_ready or vision_engine."""
        print(f"[AR OVERLAY DEBUG] Received {len(annotations)} annotations for rendering.")
        if not annotations:
            self.clear_annotations()
            return

        if self.anim.state() == QPropertyAnimation.State.Running:
            self.anim.stop()

        self._sync_geometry()
        self.active_annotations = list(annotations)
        self.setWindowOpacity(1.0)
        self.show()
        self.raise_()
        self.update()
        self.repaint()
        print(f"[AR OVERLAY DEBUG] Overlay geometry: {self.geometry().getRect()}, visible={self.isVisible()}, opacity={self.windowOpacity()}")

        # Reset and restart 10-second auto-dismiss timer
        self.fade_timer.stop()
        self.fade_timer.start(10000)

    def show_annotations(self, annotations: List[Dict[str, Any]]):
        """Alias for display_annotations."""
        self.display_annotations(annotations)

    def clear_annotations(self):
        """Immediately clears all annotations and hides overlay."""
        self.fade_timer.stop()
        if self.anim.state() == QPropertyAnimation.State.Running:
            self.anim.stop()
        self.active_annotations = []
        self.setWindowOpacity(0.0)
        self.update()
        self.hide()

    def start_fadeout(self):
        """Starts 500ms smooth fadeout animation."""
        self.anim.stop()
        self.anim.setStartValue(self.windowOpacity())
        self.anim.setEndValue(0.0)
        self.anim.start()

    def _on_fade_finished(self):
        """Hides overlay after fadeout completes."""
        self.active_annotations = []
        self.hide()

    @property
    def annotations(self) -> List[Dict[str, Any]]:
        return self.active_annotations

    @annotations.setter
    def annotations(self, value: List[Dict[str, Any]]):
        self.active_annotations = value

    def paintEvent(self, event):
        if not self.active_annotations or self.windowOpacity() <= 0.001:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        sw = self.width()
        sh = self.height()

        for item in self.active_annotations:
            box = item.get("box_2d", [0, 0, 0, 0])
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            # 1. Map normalized grid (0-1000) to display pixels
            rx = int((xmin / 1000.0) * sw)
            ry = int((ymin / 1000.0) * sh)
            rw = max(20, int(((xmax - xmin) / 1000.0) * sw))
            rh = max(20, int(((ymax - ymin) / 1000.0) * sh))

            color_hex = item.get("color", "#3b82f6")
            base_color = QColor(color_hex)
            if not base_color.isValid():
                base_color = QColor("#3b82f6")

            # 2. Draw Glowing Target Bounding Box
            glow_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 45)
            painter.setPen(QPen(glow_color, 6.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rx - 2, ry - 2, rw + 4, rh + 4, 8, 8)

            painter.setPen(QPen(base_color, 2.0))
            painter.setBrush(QBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), 25)))
            painter.drawRoundedRect(rx, ry, rw, rh, 8, 8)

            # 3. Dynamic Card Geometry Calculation (Placed directly above box, flipped below if near top)
            card_w = 240
            card_h = 68

            # Horizontally center card relative to bounding box
            card_x = rx + (rw // 2) - (card_w // 2)
            # Clamp horizontally within screen padding
            card_x = max(12, min(card_x, sw - card_w - 12))

            # Place above box; flip below if too close to top edge
            if ry > card_h + 16:
                card_y = ry - card_h - 8
            else:
                card_y = ry + rh + 8

            # 4. Render Dark Glass Card (Matching HUDContextCard style)
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1.0))
            painter.setBrush(QBrush(QColor(18, 20, 26, 230)))
            painter.drawRoundedRect(card_x, card_y, card_w, card_h, 10, 10)

            # Accent color pill indicator on card
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(base_color))
            painter.drawRoundedRect(card_x + 10, card_y + 11, 4, 14, 2, 2)

            # Title Text
            label_text = item.get("label", "")
            if label_text:
                painter.setPen(QColor(255, 255, 255, 245))
                painter.setFont(QFont("-apple-system", 11, QFont.Weight.Bold))
                painter.drawText(card_x + 20, card_y + 22, label_text)

            # Description Subtext
            desc_text = item.get("text", "")
            if desc_text:
                painter.setPen(QColor(161, 161, 170, 230))
                painter.setFont(QFont("-apple-system", 10, QFont.Weight.Normal))
                text_rect = QRectF(card_x + 10, card_y + 28, card_w - 20, 36)
                painter.drawText(text_rect, int(Qt.TextFlag.TextWordWrap), desc_text)


# Compatibility alias
ARAnnotationOverlay = AnnotationOverlay
