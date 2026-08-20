"""
Monochrome Glassmorphic On-Screen HUD Reticle Overlay for FRIDAY.
Pure white, silver, and translucent charcoal aesthetics (strictly zero neon).
Configured for macOS Fullscreen Auxiliary Spaces & High Window Levels.
"""

import math
import time
from typing import Optional, List, Tuple

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QRadialGradient,
    QPainterPath,
    QGuiApplication,
)

from gesture_recognizer import GestureState, GestureData
import permissions


class ClickRipple:
    """Expanding subtle zinc shockwave ring for click feedback."""

    def __init__(self, x: float, y: float, color: QColor = QColor(228, 228, 231), max_radius: float = 38.0, duration: float = 0.26):
        self.x = x
        self.y = y
        self.color = color
        self.max_radius = max_radius
        self.duration = duration
        self.start_time = time.time()

    @property
    def is_alive(self) -> bool:
        return (time.time() - self.start_time) < self.duration

    def draw(self, painter: QPainter):
        elapsed = time.time() - self.start_time
        progress = max(0.0, min(1.0, elapsed / self.duration))
        radius = progress * self.max_radius
        alpha = int(200 * (1.0 - progress))

        color = QColor(self.color)
        color.setAlpha(alpha)

        pen = QPen(color, max(1.0, 2.2 * (1.0 - progress)))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(self.x, self.y), radius, radius)


class TransparentOverlay(QWidget):
    """Fullscreen transparent, click-through HUD reticle overlay with monochrome palette."""

    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        screen_geo = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(screen_geo)

        # Visual settings
        self.show_reticle = True
        self.show_pinch_meter = True

        # Current State
        self.current_data: Optional[GestureData] = None
        self.ripples: List[ClickRipple] = []

        # Smooth reticle animation
        self.anim_angle = 0.0
        self.prev_state = GestureState.NONE

        # 60 FPS Repaint Timer
        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.update_animation)
        self.render_timer.start(16)

    def showEvent(self, event):
        super().showEvent(event)
        permissions.setup_macos_fullscreen_overlay(self)

    def closeEvent(self, event):
        if hasattr(self, 'render_timer'):
            self.render_timer.stop()
        super().closeEvent(event)

    def update_settings(self, settings: dict):
        """Updates overlay visual preferences."""
        if "show_reticle" in settings:
            self.show_reticle = bool(settings["show_reticle"])
        if "show_pinch_meter" in settings:
            self.show_pinch_meter = bool(settings["show_pinch_meter"])
        self.update()

    def update_gesture_data(self, data: GestureData):
        """Receives new gesture tracking data from vision engine."""
        if data.state == GestureState.CLICK and self.prev_state != GestureState.CLICK:
            self.ripples.append(ClickRipple(data.screen_x, data.screen_y, QColor(228, 228, 231), 40.0))
        elif data.state == GestureState.DOUBLE_CLICK and self.prev_state != GestureState.DOUBLE_CLICK:
            self.ripples.append(ClickRipple(data.screen_x, data.screen_y, QColor(212, 212, 216), 50.0))
        elif data.state == GestureState.RIGHT_CLICK and self.prev_state != GestureState.RIGHT_CLICK:
            self.ripples.append(ClickRipple(data.screen_x, data.screen_y, QColor(212, 212, 216), 44.0))

        self.prev_state = data.state
        self.current_data = data

    def update_animation(self):
        """Advances reticle rotation and ripple timers, then repaints."""
        self.anim_angle = (self.anim_angle + 1.8) % 360.0
        self.ripples = [r for r in self.ripples if r.is_alive]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Draw Active Ripples
        for ripple in self.ripples:
            ripple.draw(painter)

        # 2. Draw Minimalist Monochrome Reticle if tracking is active
        if self.current_data and self.current_data.is_tracking and self.show_reticle:
            cx = self.current_data.screen_x
            cy = self.current_data.screen_y
            state = self.current_data.state
            pinch = self.current_data.pinch_progress

            self._draw_reticle(painter, cx, cy, state, pinch)

        painter.end()

    def _draw_reticle(self, painter: QPainter, cx: float, cy: float, state: GestureState, pinch: float):
        """Draws subtle, minimalist monochrome HUD reticle at cursor position."""
        main_color = QColor(228, 228, 231, 210)
        glow_color = QColor(228, 228, 231, 28)

        # 1. Soft Subtle Outer Glow
        glow_grad = QRadialGradient(QPointF(cx, cy), 22)
        glow_grad.setColorAt(0.0, glow_color)
        glow_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(glow_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 22, 22)

        # 2. Outer Rotating Reticle Ring
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.anim_angle)

        pen = QPen(main_color, 1.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        r_outer = 13.0
        for i in range(4):
            start_angle = i * 90 + 12
            span_angle = 66
            painter.drawArc(QRectF(-r_outer, -r_outer, r_outer * 2, r_outer * 2), start_angle * 16, span_angle * 16)

        painter.restore()

        # 3. Inner Aim Ring
        pen_inner = QPen(QColor(212, 212, 216, 160), 1.0)
        painter.setPen(pen_inner)
        painter.drawEllipse(QPointF(cx, cy), 5, 5)

        # 4. Center Dot
        painter.setBrush(QBrush(main_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 2.0, 2.0)

        # 5. Pinch Progress Radial Ring
        if self.show_pinch_meter and pinch > 0.05:
            pinch_r = 9.5
            pinch_pen = QPen(QColor(228, 228, 231, 220), 1.8)
            pinch_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pinch_pen)
            start_deg = 90
            span_deg = -int(pinch * 360)
            painter.drawArc(QRectF(cx - pinch_r, cy - pinch_r, pinch_r * 2, pinch_r * 2), start_deg * 16, span_deg * 16)

        # 6. Scroll Chevrons
        if state == GestureState.SCROLLING:
            painter.setPen(QPen(QColor(212, 212, 216, 200), 1.5))
            painter.drawLine(int(cx - 4), int(cy - 16), int(cx), int(cy - 21))
            painter.drawLine(int(cx), int(cy - 21), int(cx + 4), int(cy - 16))
            painter.drawLine(int(cx - 4), int(cy + 16), int(cx), int(cy + 21))
            painter.drawLine(int(cx), int(cy + 21), int(cx + 4), int(cy + 16))
