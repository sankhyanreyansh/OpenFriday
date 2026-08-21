"""
AR Visual Screen Annotation Canvas for FRIDAY.
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

            # Map normalized grid to actual display pixels
            rx = int((xmin / 1000.0) * sw)
            ry = int((ymin / 1000.0) * sh)
            rw = max(4, int(((xmax - xmin) / 1000.0) * sw))
            rh = max(4, int(((ymax - ymin) / 1000.0) * sh))

            color_hex = item.get("color", "#38bdf8")
            base_color = QColor(color_hex)
            if not base_color.isValid():
                base_color = QColor("#38bdf8")

            # 1. Draw Glowing Bounding Box (Uses custom annotation color)
            glow_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 45)
            glow_pen = QPen(glow_color, 6.0)
            painter.setPen(glow_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rx - 2, ry - 2, rw + 4, rh + 4, 8, 8)

            pen = QPen(base_color, 2.0)
            painter.setPen(pen)
            fill_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 20)
            painter.setBrush(QBrush(fill_color))
            painter.drawRoundedRect(rx, ry, rw, rh, 8, 8)

            # 2. Draw Floating Glass Callout Card (Neutral dark glass matching HUDContextCard - NO colored borders/bars)
            label_title = item.get("label", "")
            desc_text = item.get("text", "")
            if not label_title and not desc_text:
                continue

            # Position card directly above or below the bounding box
            card_w = 230
            card_h = 66
            card_x = max(10, min(rx + (rw // 2) - (card_w // 2), sw - card_w - 10))
            card_y = ry - 75 if ry > 90 else ry + rh + 10

            # Pure neutral dark glass matching HUDContextCard
            painter.setPen(QPen(QColor(255, 255, 255, 30), 1.0))
            painter.setBrush(QBrush(QColor(18, 20, 26, 225)))
            painter.drawRoundedRect(card_x, card_y, card_w, card_h, 12, 12)

            # Title Label
            if label_title:
                painter.setPen(QColor(255, 255, 255, 240))
                title_font = QFont("-apple-system", 11, QFont.Weight.Bold)
                painter.setFont(title_font)
                painter.drawText(card_x + 12, card_y + 18, label_title)

            # Description Text
            if desc_text:
                painter.setPen(QColor(161, 161, 170, 230))
                body_font = QFont("-apple-system", 10, QFont.Weight.Normal)
                painter.setFont(body_font)
                text_rect = QRectF(card_x + 12, card_y + 24, card_w - 24, 38)
                painter.drawText(text_rect, int(Qt.TextFlag.TextWordWrap), desc_text)


# Compatibility alias
ARAnnotationOverlay = AnnotationOverlay
