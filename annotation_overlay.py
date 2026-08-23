"""
AR Visual Screen Annotation Canvas for Open FRIDAY.
Renders clean glowing bounding boxes and floating glassmorphic callout cards
anchored directly to on-screen components, diagrams, circuits, and code.
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
    QGuiApplication,
)
import permissions


class AnnotationOverlay(QWidget):
    """Fullscreen transparent, click-through AR canvas for visual annotations."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(None)
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
        self.dismiss_timer = self.fade_timer

        # Opacity Fadeout Animation
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(450)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.anim.finished.connect(self._on_fade_finished)
        self.fade_anim = self.anim

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
        """Slot to render a list of visual bounding boxes and callout cards."""
        print(f"[AR OVERLAY] Received {len(annotations)} annotations for rendering.")
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

        # Reset and restart 10-second auto-dismiss timer
        self.fade_timer.stop()
        self.fade_timer.start(10000)

    def show_annotations(self, annotations: List[Dict[str, Any]]):
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
        """Starts smooth fadeout animation."""
        if not self.isVisible() or self.windowOpacity() <= 0.01:
            return
        self.anim.stop()
        self.anim.setStartValue(self.windowOpacity())
        self.anim.setEndValue(0.0)
        self.anim.start()

    def _on_fade_finished(self):
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

        sw = float(self.width())
        sh = float(self.height())

        title_font = QFont("Helvetica Neue", 11, QFont.Weight.Bold)
        body_font = QFont("Helvetica Neue", 10, QFont.Weight.Normal)
        body_metrics = QFontMetrics(body_font)

        occupied_card_rects: List[QRectF] = []

        for item in self.active_annotations:
            box = item.get("box_2d", [0, 0, 0, 0])
            if not box or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            # 1. Map normalized grid (0-1000) to display pixels
            rx = (xmin / 1000.0) * sw
            ry = (ymin / 1000.0) * sh
            rw = max(16.0, ((xmax - xmin) / 1000.0) * sw)
            rh = max(16.0, ((ymax - ymin) / 1000.0) * sh)
            target_rect = QRectF(rx, ry, rw, rh)

            color_hex = item.get("color", "#3b82f6")
            base_color = QColor(color_hex)
            if not base_color.isValid():
                base_color = QColor("#3b82f6")

            # 2. Draw Glowing Target Bounding Box
            glow_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 45)
            painter.setPen(QPen(glow_color, 6.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(rx - 2, ry - 2, rw + 4, rh + 4), 8.0, 8.0)

            painter.setPen(QPen(base_color, 2.0))
            painter.setBrush(QBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), 25)))
            painter.drawRoundedRect(target_rect, 8.0, 8.0)

            # 3. Dynamic Card Geometry Calculation (FontMetrics based)
            card_w = 260.0
            label_text = str(item.get("label", ""))
            desc_text = str(item.get("text", ""))

            content_h = 16.0
            if label_text:
                content_h += 16.0
            if desc_text:
                text_bounding = body_metrics.boundingRect(
                    0, 0, int(card_w - 24.0), 1000, int(Qt.TextFlag.TextWordWrap), desc_text
                )
                content_h += max(18.0, float(text_bounding.height())) + 8.0

            card_h = max(50.0, min(140.0, content_h + 10.0))

            # Initial placement: centered above box
            card_x = rx + (rw / 2.0) - (card_w / 2.0)
            card_x = max(12.0, min(card_x, sw - card_w - 12.0))

            if ry > card_h + 16.0:
                card_y = ry - card_h - 10.0
            else:
                card_y = ry + rh + 10.0

            # Collision avoidance pass against previously placed cards
            card_rect = QRectF(card_x, card_y, card_w, card_h)
            for prev_rect in occupied_card_rects:
                if card_rect.intersects(prev_rect):
                    # Shift vertically or horizontally
                    if card_y < prev_rect.y():
                        card_y = prev_rect.y() - card_h - 8.0
                    else:
                        card_y = prev_rect.y() + prev_rect.height() + 8.0
                    card_y = max(12.0, min(card_y, sh - card_h - 12.0))
                    card_rect = QRectF(card_x, card_y, card_w, card_h)

            occupied_card_rects.append(card_rect)

            # 4. Render Dark Glassmorphic Card
            painter.setPen(QPen(QColor(255, 255, 255, 32), 1.0))
            painter.setBrush(QBrush(QColor(16, 18, 24, 235)))
            painter.drawRoundedRect(card_rect, 10.0, 10.0)

            # Accent color pill indicator on card
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(base_color))
            painter.drawRoundedRect(QRectF(card_rect.x() + 10, card_rect.y() + 12, 4, 14), 2.0, 2.0)

            # Title Text
            if label_text:
                painter.setPen(QColor(255, 255, 255, 245))
                painter.setFont(title_font)
                painter.drawText(int(card_rect.x() + 20), int(card_rect.y() + 24), label_text)

            # Description Subtext
            if desc_text:
                painter.setPen(QColor(212, 212, 216, 235))
                painter.setFont(body_font)
                text_rect = QRectF(card_rect.x() + 10, card_rect.y() + 32, card_w - 20, card_h - 38)
                painter.drawText(text_rect, int(Qt.TextFlag.TextWordWrap), desc_text)

        painter.end()


# Compatibility alias
ARAnnotationOverlay = AnnotationOverlay
