"""
Transparent Screen-Wide Overlay for Open FRIDAY.
Renders high-precision cursor reticle, dynamic pinch gauges, click ripple animations,
and the customizable N-sector Radial Shortcut Menu at 60 FPS.
"""

import math
import time
from typing import List, Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QPainterPath,
    QRadialGradient,
    QGuiApplication,
)

from src.control import permissions
from src.control.radial_menu import RADIAL_ACTIONS
from src.vision.gesture_recognizer import GestureState, GestureData


class ClickRipple:
    """Animated expanding ripple triggered on click events."""

    def __init__(
        self,
        x: float,
        y: float,
        color: QColor = QColor(228, 228, 231),
        max_radius: float = 38.0,
        duration: float = 0.26,
    ):
        self.x = x
        self.y = y
        self.color = color
        self.max_radius = max_radius
        self.duration = duration
        self.start_time = time.time()

    def is_alive(self) -> bool:
        return (time.time() - self.start_time) < self.duration

    def draw(self, painter: QPainter):
        elapsed = time.time() - self.start_time
        progress = min(1.0, elapsed / self.duration)

        radius = progress * self.max_radius
        alpha = int((1.0 - progress) * 200)

        ripple_color = QColor(self.color)
        ripple_color.setAlpha(alpha)

        pen = QPen(ripple_color, max(1.0, 2.0 * (1.0 - progress)))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(self.x, self.y), radius, radius)


class TransparentOverlay(QWidget):
    """Fullscreen transparent, click-through overlay window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open FRIDAY - Visual Overlay")

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        screen = QGuiApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())
            self.screen_width = screen.geometry().width()
            self.screen_height = screen.geometry().height()
        else:
            self.screen_width = 1470
            self.screen_height = 956
            self.resize(self.screen_width, self.screen_height)

        # State data
        self.cursor_x = self.screen_width / 2.0
        self.cursor_y = self.screen_height / 2.0
        self.is_tracking = False
        self.current_state = GestureState.NONE
        self.pinch_progress = 0.0
        self.is_modifier_active = False
        self.active_radial_sector: Optional[str] = None
        self.is_dictating = False

        # Settings
        self.show_reticle = True
        self.show_pinch_meter = True

        # Click ripples & animation timer
        self.ripples: List[ClickRipple] = []
        self.prev_state = GestureState.NONE
        self._needs_repaint = True

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(16)  # ~60 FPS

    def showEvent(self, event):
        super().showEvent(event)
        permissions.setup_macos_fullscreen_overlay(self)

    def closeEvent(self, event):
        self.anim_timer.stop()
        super().closeEvent(event)

    def update_settings(self, settings: dict):
        if "show_reticle" in settings:
            self.show_reticle = bool(settings["show_reticle"])
        if "show_pinch_meter" in settings:
            self.show_pinch_meter = bool(settings["show_pinch_meter"])
        self._needs_repaint = True

    def update_gesture_data(self, data: GestureData):
        """Updates internal state and triggers ripple animations on clicks."""
        coords_changed = (
            abs(self.cursor_x - data.screen_x) > 0.5 or
            abs(self.cursor_y - data.screen_y) > 0.5
        )
        is_dictating = getattr(data, "is_dictating", False)
        state_changed = (
            self.current_state != data.state or
            self.is_tracking != data.is_tracking or
            self.is_modifier_active != data.is_modifier_active or
            self.active_radial_sector != data.radial_sector or
            self.is_dictating != is_dictating or
            abs(self.pinch_progress - data.pinch_progress) > 0.02
        )
        if coords_changed or state_changed:
            self._needs_repaint = True

        self.cursor_x = data.screen_x
        self.cursor_y = data.screen_y
        self.is_tracking = data.is_tracking
        self.current_state = data.state
        self.pinch_progress = data.pinch_progress
        self.is_modifier_active = data.is_modifier_active
        self.active_radial_sector = data.radial_sector
        self.is_dictating = is_dictating

        # Trigger ripples on click transitions
        if self.current_state != self.prev_state:
            if self.current_state == GestureState.CLICK:
                self.ripples.append(ClickRipple(self.cursor_x, self.cursor_y, QColor(244, 244, 245), 36.0, 0.24))
            elif self.current_state == GestureState.DOUBLE_CLICK:
                self.ripples.append(ClickRipple(self.cursor_x, self.cursor_y, QColor(255, 255, 255), 48.0, 0.32))
            elif self.current_state == GestureState.RIGHT_CLICK:
                self.ripples.append(ClickRipple(self.cursor_x, self.cursor_y, QColor(212, 212, 216), 40.0, 0.28))
            self.prev_state = self.current_state
            self._needs_repaint = True

    def update_animation(self):
        """Advances ripple animations and repaints if dirty or ripples are active."""
        if self.ripples:
            self.ripples = [r for r in self.ripples if r.is_alive()]
            self.update()
        elif self._needs_repaint:
            self._needs_repaint = False
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Click ripples
        for ripple in self.ripples:
            ripple.draw(painter)

        # 2. Radial Shortcut Menu (Center screen)
        if self.current_state == GestureState.RADIAL_MENU:
            cx = self.screen_width / 2.0
            cy = self.screen_height / 2.0
            self._draw_radial_menu(
                painter,
                cx,
                cy,
                self.active_radial_sector,
                self.cursor_x,
                self.cursor_y,
                self.pinch_progress,
            )

        # 3. Pointer Reticle
        if self.is_tracking and self.show_reticle and self.current_state != GestureState.RADIAL_MENU:
            self._draw_reticle(painter, self.cursor_x, self.cursor_y, self.current_state, self.pinch_progress)

        painter.end()

    def _draw_radial_menu(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        active_sector: Optional[str],
        cursor_x: float,
        cursor_y: float,
        pinch: float,
    ):
        """Draws a customizable N-sector radial shortcut wheel dynamically."""
        if not RADIAL_ACTIONS:
            return

        r_out = 160.0
        r_in = 45.0
        n_sectors = len(RADIAL_ACTIONS)
        sweep_deg = 360.0 / n_sectors

        # 1. Backdrop Glow
        bg_glow = QRadialGradient(QPointF(cx, cy), r_out + 40)
        bg_glow.setColorAt(0.0, QColor(0, 0, 0, 160))
        bg_glow.setColorAt(0.7, QColor(0, 0, 0, 90))
        bg_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(bg_glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), r_out + 40, r_out + 40)

        # 2. Outer Rim
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1.2))
        painter.drawEllipse(QPointF(cx, cy), r_out + 2, r_out + 2)

        # 3. Dynamic Sectors
        for i, action in enumerate(RADIAL_ACTIONS):
            sec_id = action["id"]
            label = action.get("label", sec_id)
            symbol = action.get("symbol", "")

            # Cartesian angle: index 0 is East (0 deg), index 1 is North (90 deg), etc.
            mid_deg = i * sweep_deg
            start_deg = mid_deg - sweep_deg / 2.0
            is_active = (active_sector == sec_id)

            path = QPainterPath()
            path.arcMoveTo(cx - r_out, cy - r_out, r_out * 2, r_out * 2, start_deg)
            path.arcTo(cx - r_out, cy - r_out, r_out * 2, r_out * 2, start_deg, sweep_deg)
            path.arcTo(cx - r_in, cy - r_in, r_in * 2, r_in * 2, start_deg + sweep_deg, -sweep_deg)
            path.closeSubpath()

            if is_active:
                painter.setBrush(QBrush(QColor(113, 113, 122, 120)))
                painter.setPen(QPen(QColor(244, 244, 245, 235), 2.0))
            else:
                painter.setBrush(QBrush(QColor(18, 20, 26, 215)))
                painter.setPen(QPen(QColor(255, 255, 255, 32), 1.0))

            painter.drawPath(path)

            # Draw Quadrant Symbols & Labels at midpoint
            rad = math.radians(mid_deg)
            r_mid = 106.0
            tx = cx + r_mid * math.cos(rad)
            ty = cy - r_mid * math.sin(rad)

            sym_color = QColor(244, 244, 245) if is_active else QColor(161, 161, 170, 200)
            lbl_color = QColor(244, 244, 245) if is_active else QColor(140, 140, 150, 180)

            # Symbol
            painter.setPen(sym_color)
            sym_font = QFont("Helvetica Neue", 15, QFont.Weight.Bold if is_active else QFont.Weight.Medium)
            painter.setFont(sym_font)
            painter.drawText(QRectF(tx - 40, ty - 22, 80, 22), Qt.AlignmentFlag.AlignCenter, symbol)

            # Label
            painter.setPen(lbl_color)
            lbl_font = QFont("Helvetica Neue", 9, QFont.Weight.DemiBold if is_active else QFont.Weight.Normal)
            lbl_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.5)
            painter.setFont(lbl_font)
            painter.drawText(QRectF(tx - 45, ty + 2, 90, 18), Qt.AlignmentFlag.AlignCenter, label)

        # 4. Center Hub
        painter.setBrush(QBrush(QColor(14, 16, 22, 245)))
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1.2))
        painter.drawEllipse(QPointF(cx, cy), r_in, r_in)

        painter.setPen(QColor(161, 161, 170, 180))
        center_font = QFont("Helvetica Neue", 9, QFont.Weight.DemiBold)
        center_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        painter.setFont(center_font)
        painter.drawText(QRectF(cx - 40, cy - 10, 80, 20), Qt.AlignmentFlag.AlignCenter, "SHORTCUTS")

        # 5. Pointer Aim Line
        dx = cursor_x - cx
        dy = cursor_y - cy
        dist = math.hypot(dx, dy)

        if dist > 5.0:
            norm_x = dx / dist
            norm_y = dy / dist
            line_len = min(dist, r_out + 10)
            end_x = cx + norm_x * line_len
            end_y = cy + norm_y * line_len

            aim_pen = QPen(QColor(255, 255, 255, 160 if active_sector else 70), 1.5)
            aim_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(aim_pen)
            painter.drawLine(QPointF(cx, cy), QPointF(end_x, end_y))

        # Aim dot
        painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
        painter.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
        painter.drawEllipse(QPointF(cursor_x, cursor_y), 4.5, 4.5)

        # Pinch Arc
        if pinch > 0.05:
            pinch_r = 10.5
            pinch_pen = QPen(QColor(255, 255, 255, 240), 2.0)
            pinch_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pinch_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(
                QRectF(cursor_x - pinch_r, cursor_y - pinch_r, pinch_r * 2, pinch_r * 2),
                90 * 16,
                -int(pinch * 360) * 16,
            )

    def _draw_reticle(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        state: GestureState,
        pinch: float,
    ):
        """Draws the dynamic on-screen cursor reticle."""
        if state == GestureState.DRAGGING:
            r = 11.0
            color = QColor(255, 255, 255, 230)
            fill_color = QColor(255, 255, 255, 120)
        elif state == GestureState.RIGHT_CLICK or (self.is_modifier_active and state == GestureState.CLICK):
            r = 10.0
            color = QColor(228, 228, 231, 240)
            fill_color = QColor(228, 228, 231, 90)
        elif state in (GestureState.CLICK, GestureState.DOUBLE_CLICK):
            r = 12.0
            color = QColor(255, 255, 255, 255)
            fill_color = QColor(255, 255, 255, 160)
        elif state == GestureState.SCROLLING:
            r = 9.0
            color = QColor(212, 212, 216, 220)
            fill_color = QColor(212, 212, 216, 40)
        elif state == GestureState.ZOOM:
            r = 13.0
            color = QColor(255, 255, 255, 240)
            fill_color = QColor(255, 255, 255, 80)
        else:
            r = 8.5
            color = QColor(212, 212, 216, 200)
            fill_color = QColor(212, 212, 216, 30)

        # Center dot
        painter.setBrush(QBrush(fill_color))
        painter.setPen(QPen(color, 1.5))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Pinch progress ring
        if self.show_pinch_meter and pinch > 0.04 and state not in (GestureState.CLICK, GestureState.DOUBLE_CLICK):
            ring_r = r + 4.5
            ring_pen = QPen(QColor(255, 255, 255, 220), 2.0)
            ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(
                QRectF(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2),
                90 * 16,
                -int(pinch * 360) * 16,
            )

        # Dictation recording indicator badge
        if self.is_dictating:
            mic_dot_x = cx + r + 5.0
            mic_dot_y = cy - r
            painter.setBrush(QBrush(QColor(239, 68, 68, 230)))
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
            painter.drawEllipse(QPointF(mic_dot_x, mic_dot_y), 3.5, 3.5)
