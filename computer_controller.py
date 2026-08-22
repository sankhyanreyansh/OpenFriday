"""
Native macOS Computer Controller for FRIDAY.
Provides high-speed GUI automation using native Quartz CoreGraphics, AppleScript,
and in-memory scaled screenshots with normalized coordinate mapping.
"""

import subprocess
import time
import io
import base64
from typing import Tuple
from PIL import ImageGrab
import Quartz.CoreGraphics as CG

from grid_overlay import draw_visual_coordinate_grid


class MacComputerController:
    """Controls mouse, keyboard, and screen capture on macOS natively via Quartz and AppleScript."""

    def __init__(self):
        main_display = CG.CGMainDisplayID()
        self.logical_width = CG.CGDisplayPixelsWide(main_display)
        self.logical_height = CG.CGDisplayPixelsHigh(main_display)
        self.screen_width = self.logical_width
        self.screen_height = self.logical_height
        self.last_capture_width = self.logical_width
        self.last_capture_height = self.logical_height

    def capture_screen_base64(self, apply_grid: bool = True) -> Tuple[str, int, int]:
        """
        Captures the primary display into a compressed PNG base64 string,
        strictly preserving the native 1:1 screen aspect ratio without distortion.
        Optionally overlays a 0-1000 coordinate grid for VLM spatial localization.
        Returns:
            (base64_str, current_image_width, current_image_height)
        """
        screenshot = ImageGrab.grab()
        if apply_grid:
            screenshot = draw_visual_coordinate_grid(screenshot)

        raw_w, raw_h = screenshot.size

        # Scale to max 1280px width while strictly preserving aspect ratio
        if raw_w > 1280:
            scale = 1280.0 / float(raw_w)
            target_w = 1280
            target_h = int(round(raw_h * scale))
            resized_img = screenshot.resize((target_w, target_h))
        else:
            target_w = raw_w
            target_h = raw_h
            resized_img = screenshot

        self.last_capture_width = target_w
        self.last_capture_height = target_h

        buf = io.BytesIO()
        resized_img.save(buf, format="PNG", optimize=True)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64_str, target_w, target_h

    def map_coordinates(self, norm_x: int, norm_y: int) -> Tuple[int, int]:
        """Maps normalized (0-1000) coordinates to macOS logical screen points."""
        logical_x = int(round((norm_x / 1000.0) * self.screen_width))
        logical_y = int(round((norm_y / 1000.0) * self.screen_height))
        clamped_x = max(0, min(logical_x, self.screen_width - 1))
        clamped_y = max(0, min(logical_y, self.screen_height - 1))
        return clamped_x, clamped_y

    def click(self, x: int, y: int, button: str = "left", double: bool = False) -> str:
        """
        Direct Quartz click mapped from a normalized 0-1000 integer grid to macOS logical screen points.
        (0, 0) is top-left, (1000, 1000) is bottom-right.
        Posts explicit mouse movement first so macOS window/element focus activates.
        """
        target_x, target_y = self.map_coordinates(x, y)
        print(f"[COMPUTER AGENT DEBUG] Click: Norm({x}, {y}) -> Logical({target_x}, {target_y}) on ({self.screen_width}x{self.screen_height})")

        point = CG.CGPoint(x=target_x, y=target_y)

        # 1. Move cursor to target position
        move_event = CG.CGEventCreateMouseEvent(None, CG.kCGEventMouseMoved, point, CG.kCGMouseButtonLeft)
        CG.CGEventPost(CG.kCGHIDEventTap, move_event)
        time.sleep(0.05)

        # 2. Post Mouse Down & Up
        btn_lower = button.lower()
        down_type = CG.kCGEventLeftMouseDown if btn_lower == "left" else CG.kCGEventRightMouseDown
        up_type = CG.kCGEventLeftMouseUp if btn_lower == "left" else CG.kCGEventRightMouseUp
        btn = CG.kCGMouseButtonLeft if btn_lower == "left" else CG.kCGMouseButtonRight

        down_event = CG.CGEventCreateMouseEvent(None, down_type, point, btn)
        up_event = CG.CGEventCreateMouseEvent(None, up_type, point, btn)

        CG.CGEventPost(CG.kCGHIDEventTap, down_event)
        time.sleep(0.05)
        CG.CGEventPost(CG.kCGHIDEventTap, up_event)

        if double:
            time.sleep(0.08)
            CG.CGEventPost(CG.kCGHIDEventTap, down_event)
            time.sleep(0.05)
            CG.CGEventPost(CG.kCGHIDEventTap, up_event)
            return f"Double-clicked {button} at logical ({target_x}, {target_y}) [norm: ({x}, {y})]"

        return f"Clicked {button} at logical ({target_x}, {target_y}) [norm: ({x}, {y})]"

    def type_text(self, text: str) -> str:
        """Instant zero-latency text insertion via macOS pbcopy and paste."""
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE, close_fds=True)
        p.communicate(text.encode("utf-8"))
        self.run_applescript('tell application "System Events" to keystroke "v" using command down')
        time.sleep(0.05)
        return f"Typed text: {text[:40]}{'...' if len(text) > 40 else ''}"

    def key_press(self, key_name: str) -> str:
        """Dispatches native macOS keyboard events."""
        key_clean = key_name.lower().strip()
        key_map = {
            "enter": "key code 36",
            "return": "key code 36",
            "tab": "key code 48",
            "space": "key code 49",
            "escape": "key code 53",
            "esc": "key code 53",
            "delete": "key code 51",
            "backspace": "key code 51",
            "up": "key code 126",
            "down": "key code 125",
            "left": "key code 123",
            "right": "key code 124",
            "cmd+a": 'keystroke "a" using command down',
            "cmd+c": 'keystroke "c" using command down',
            "cmd+v": 'keystroke "v" using command down',
            "cmd+w": 'keystroke "w" using command down',
            "cmd+q": 'keystroke "q" using command down',
            "cmd+t": 'keystroke "t" using command down',
            "cmd+f": 'keystroke "f" using command down',
            "cmd+space": 'key code 49 using command down',
        }
        code = key_map.get(key_clean)
        if code:
            self.run_applescript(f'tell application "System Events" to {code}')
        else:
            self.run_applescript(f'tell application "System Events" to keystroke "{key_name}"')
        time.sleep(0.05)
        return f"Pressed key: {key_name}"

    def scroll(self, dy: int = 0, dx: int = 0) -> str:
        """Scrolls vertically (dy) or horizontally (dx) via Quartz."""
        scroll_event = CG.CGEventCreateScrollWheelEvent2(
            None,
            CG.kCGScrollEventUnitPixel,
            2,        # wheel count
            int(dy),  # vertical scroll delta
            int(dx),  # horizontal scroll delta
            0         # wheel3
        )
        CG.CGEventPost(CG.kCGHIDEventTap, scroll_event)
        time.sleep(0.05)
        return f"Scrolled dy={dy}, dx={dx}"

    def run_applescript(self, script: str) -> str:
        """Executes an AppleScript command via osascript."""
        try:
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
            return res.stdout.strip()
        except Exception as e:
            return f"AppleScript error: {e}"

