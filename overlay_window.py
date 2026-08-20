"""
Monochrome Glassmorphic On-Screen HUD Reticle Overlay for FRIDAY.
Pure white, silver, and translucent charcoal aesthetics (strictly zero neon).
Configured for macOS Fullscreen Auxiliary Spaces & High Window Levels.
"""

import math
import time
from typing import Optional, List, Tuple

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal
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

    dismiss_radial_requested = pyqtSignal()

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

    def mousePressEvent(self, event):
        """Physical mouse click closes the radial menu if open."""
        if self.current_data and self.current_data.state == GestureState.RADIAL_MENU:
            self.dismiss_radial_requested.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

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
        elif data.state == GestureState.RADIAL_MENU and data.nav_action in ("ENTER", "NEW_TAB", "CLOSE_TAB", "ESCAPE"):
            # Trigger shockwave ripple on shortcut selection
            center_x = self.width() / 2.0
            center_y = self.height() / 2.0
            self.ripples.append(ClickRipple(center_x, center_y, QColor(255, 255, 255), 120.0, duration=0.35))

        # Enable click listening while in radial menu to allow physical mouse clicks to dismiss it
        if data.state == GestureState.RADIAL_MENU:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

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

        # 2. Draw Snipping Bounding Box if in SNIP_DRAG mode
        if self.current_data and self.current_data.state == GestureState.SNIP_DRAG and self.current_data.snip_box:
            self._draw_snip_selection(painter, self.current_data.snip_box)

        # 3. Draw GTA-Style Radial Shortcut Wheel if in RADIAL_MENU state
        if self.current_data and self.current_data.state == GestureState.RADIAL_MENU:
            cx = self.width() / 2.0
            cy = self.height() / 2.0
            active_sec = getattr(self.current_data, 'radial_sector', None)
            cur_x = self.current_data.screen_x if self.current_data.is_tracking else cx
            cur_y = self.current_data.screen_y if self.current_data.is_tracking else cy
            pinch = self.current_data.pinch_progress
            self._draw_radial_menu(painter, cx, cy, active_sec, cur_x, cur_y, pinch)

        # 4. Draw Minimalist Monochrome Reticle if tracking is active
        elif self.current_data and self.current_data.is_tracking and self.show_reticle:
            cx = self.current_data.screen_x
            cy = self.current_data.screen_y
            state = self.current_data.state
            pinch = self.current_data.pinch_progress

            self._draw_reticle(painter, cx, cy, state, pinch)

        painter.end()

    def _draw_snip_selection(self, painter: QPainter, snip_box: Tuple[float, float, float, float]):
        """Draws a translucent selection rectangle with dashed border and corner accents."""
        x1, y1, x2, y2 = snip_box
        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = abs(x2 - x1)
        rh = abs(y2 - y1)

        if rw < 2 or rh < 2:
            return

        rect = QRectF(rx, ry, rw, rh)

        # Translucent selection fill
        painter.setBrush(QBrush(QColor(255, 255, 255, 20)))

        # Dashed glowing border
        border_pen = QPen(QColor(244, 244, 245, 180), 1.5, Qt.PenStyle.DashLine)
        border_pen.setDashPattern([6, 4])
        painter.setPen(border_pen)
        painter.drawRect(rect)

        # Corner bracket accents
        corner_pen = QPen(QColor(255, 255, 255, 240), 2.0)
        painter.setPen(corner_pen)
        c_len = min(12.0, rw / 2.0, rh / 2.0)

        # Top-Left
        painter.drawLine(QPointF(rx, ry), QPointF(rx + c_len, ry))
        painter.drawLine(QPointF(rx, ry), QPointF(rx, ry + c_len))
        # Top-Right
        painter.drawLine(QPointF(rx + rw, ry), QPointF(rx + rw - c_len, ry))
        painter.drawLine(QPointF(rx + rw, ry), QPointF(rx + rw, ry + c_len))
        # Bottom-Left
        painter.drawLine(QPointF(rx, ry + rh), QPointF(rx + c_len, ry + rh))
        painter.drawLine(QPointF(rx, ry + rh), QPointF(rx, ry + rh - c_len))
        # Bottom-Right
        painter.drawLine(QPointF(rx + rw, ry + rh), QPointF(rx + rw - c_len, ry + rh))
        painter.drawLine(QPointF(rx + rw, ry + rh), QPointF(rx + rw, ry + rh - c_len))

        # Dimensions / Status Label tag
        tag_text = f"● SNIP CONTEXT ({int(rw)} × {int(rh)})"
        painter.setFont(QFont("Helvetica Neue", 10, QFont.Weight.DemiBold))
        tag_rect = QRectF(rx, max(10.0, ry - 22.0), max(200.0, rw), 18.0)
        painter.setPen(QColor(244, 244, 245, 220))
        painter.drawText(tag_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tag_text)


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
        """Draws a sleek, high-contrast monochrome GTA-style radial shortcut wheel at screen center."""
        r_out = 160.0
        r_in = 45.0

        # 1. Soft Backdrop Glow
        bg_glow = QRadialGradient(QPointF(cx, cy), r_out + 40)
        bg_glow.setColorAt(0.0, QColor(0, 0, 0, 160))
        bg_glow.setColorAt(0.7, QColor(0, 0, 0, 90))
        bg_glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(bg_glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), r_out + 40, r_out + 40)

        # 2. Outer Subtle Rim
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1.2))
        painter.drawEllipse(QPointF(cx, cy), r_out + 2, r_out + 2)

        # 3. 4 Radial Quadrants:
        # RIGHT (⌘T / NEW TAB), TOP (↵ / ENTER), LEFT (⌘W / CLOSE TAB), BOTTOM (⎋ / ESCAPE)
        sectors = [
            ("NEW_TAB", "NEW TAB", "⌘T", -45, 90, 0),
            ("ENTER", "ENTER", "↵", 45, 90, 90),
            ("CLOSE_TAB", "CLOSE TAB", "⌘W", 135, 90, 180),
            ("ESCAPE", "ESCAPE", "⎋", 225, 90, 270),
        ]

        for sec_id, label, sublabel, start_deg, sweep_deg, mid_deg in sectors:
            is_active = (active_sector == sec_id)

            path = QPainterPath()
            path.arcMoveTo(cx - r_out, cy - r_out, r_out * 2, r_out * 2, start_deg)
            path.arcTo(cx - r_out, cy - r_out, r_out * 2, r_out * 2, start_deg, sweep_deg)
            path.arcTo(cx - r_in, cy - r_in, r_in * 2, r_in * 2, start_deg + sweep_deg, -sweep_deg)
            path.closeSubpath()

            if is_active:
                # Subtle, Refined Zinc/Charcoal Gray Active Sector
                painter.setBrush(QBrush(QColor(113, 113, 122, 110)))
                painter.setPen(QPen(QColor(244, 244, 245, 230), 2.0))
            else:
                # Sleek Glassmorphic Inactive Sector
                painter.setBrush(QBrush(QColor(18, 20, 26, 215)))
                painter.setPen(QPen(QColor(255, 255, 255, 32), 1.0))

            painter.drawPath(path)

            # Draw Quadrant Symbols & Labels
            rad = math.radians(mid_deg)
            r_mid = 106.0
            tx = cx + r_mid * math.cos(rad)
            ty = cy - r_mid * math.sin(rad)

            sym_color = QColor(244, 244, 245) if is_active else QColor(161, 161, 170, 200)
            lbl_color = QColor(244, 244, 245) if is_active else QColor(140, 140, 150, 180)

            # Shortcut Symbol
            painter.setPen(sym_color)
            sym_font = QFont("Helvetica Neue", 16, QFont.Weight.Bold if is_active else QFont.Weight.Medium)
            painter.setFont(sym_font)
            painter.drawText(QRectF(tx - 40, ty - 20, 80, 22), Qt.AlignmentFlag.AlignCenter, sublabel)

            # Action Label
            painter.setPen(lbl_color)
            lbl_font = QFont("Helvetica Neue", 10, QFont.Weight.DemiBold if is_active else QFont.Weight.Normal)
            lbl_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
            painter.setFont(lbl_font)
            painter.drawText(QRectF(tx - 50, ty + 2, 100, 18), Qt.AlignmentFlag.AlignCenter, label)



        # 4. Center Deadzone Hub
        painter.setBrush(QBrush(QColor(14, 16, 22, 245)))
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1.2))
        painter.drawEllipse(QPointF(cx, cy), r_in, r_in)

        # Center Label
        painter.setPen(QColor(161, 161, 170, 180))
        center_font = QFont("Helvetica Neue", 9, QFont.Weight.DemiBold)
        center_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        painter.setFont(center_font)
        painter.drawText(QRectF(cx - 40, cy - 10, 80, 20), Qt.AlignmentFlag.AlignCenter, "SHORTCUTS")

        # 5. Pointer Aim Line & Position Indicator
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

        # Aim Pointer Dot at Cursor
        painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
        painter.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
        painter.drawEllipse(QPointF(cursor_x, cursor_y), 4.5, 4.5)

        # Pinch Meter Ring during Pinch Tap
        if pinch > 0.05:
            pinch_r = 10.5
            pinch_pen = QPen(QColor(255, 255, 255, 240), 2.0)
            pinch_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pinch_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(
                QRectF(cursor_x - pinch_r, cursor_y - pinch_r, pinch_r * 2, pinch_r * 2),
                90 * 16,
                -int(pinch * 360) * 16
            )

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

