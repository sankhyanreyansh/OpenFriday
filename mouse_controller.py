"""
macOS Native Mouse & Cursor Controller using Quartz CoreGraphics.
Provides low-latency, high-precision cursor movement, left/right clicks,
drag-and-drop, double clicks, and linear bidirectional scrolling directly into the macOS Window Server.
"""

import sys
import time
import subprocess
import threading
from typing import Tuple, Optional

try:
    import Quartz.CoreGraphics as CG
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0


def _run_system_keystroke(key_code: int):
    """Dispatches Control + KeyCode reliably via macOS System Events."""
    script = f'tell application "System Events" to key code {key_code} using control down'
    subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class MouseController:
    """Controls the macOS system cursor and mouse events."""

    def __init__(self):
        self.quartz_available = QUARTZ_AVAILABLE
        self.screen_width, self.screen_height = self.get_screen_size()
        self.is_mouse_down = False
        self.current_x = self.screen_width / 2.0
        self.current_y = self.screen_height / 2.0

    def get_screen_size(self) -> Tuple[int, int]:
        """Returns the primary screen width and height in points."""
        if self.quartz_available:
            main_display = CG.CGMainDisplayID()
            w = CG.CGDisplayPixelsWide(main_display)
            h = CG.CGDisplayPixelsHigh(main_display)
            return int(w), int(h)
        else:
            w, h = pyautogui.size()
            return int(w), int(h)

    def clamp(self, x: float, y: float) -> Tuple[float, float]:
        """Ensures coordinates stay within screen boundaries."""
        cx = max(0.0, min(float(x), float(self.screen_width - 1)))
        cy = max(0.0, min(float(y), float(self.screen_height - 1)))
        return cx, cy

    def move_to(self, x: float, y: float):
        """Moves cursor to (x, y). If mouse is currently down, sends drag event."""
        cx, cy = self.clamp(x, y)
        self.current_x = cx
        self.current_y = cy

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            if self.is_mouse_down:
                event = CG.CGEventCreateMouseEvent(
                    None, CG.kCGEventLeftMouseDragged, loc, CG.kCGMouseButtonLeft
                )
            else:
                event = CG.CGEventCreateMouseEvent(
                    None, CG.kCGEventMouseMoved, loc, CG.kCGMouseButtonLeft
                )
            if event:
                CG.CGEventPost(CG.kCGHIDEventTap, event)
        else:
            if self.is_mouse_down:
                pyautogui.dragTo(int(cx), int(cy), button='left', _pause=False)
            else:
                pyautogui.moveTo(int(cx), int(cy), _pause=False)

    def mouse_down(self, x: Optional[float] = None, y: Optional[float] = None):
        """Presses and holds the left mouse button."""
        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        self.is_mouse_down = True

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventLeftMouseDown, loc, CG.kCGMouseButtonLeft
            )
            if event:
                CG.CGEventPost(CG.kCGHIDEventTap, event)
        else:
            pyautogui.mouseDown(int(cx), int(cy), button='left', _pause=False)

    def mouse_up(self, x: Optional[float] = None, y: Optional[float] = None):
        """Releases the left mouse button."""
        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        self.is_mouse_down = False

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventLeftMouseUp, loc, CG.kCGMouseButtonLeft
            )
            if event:
                CG.CGEventPost(CG.kCGHIDEventTap, event)
        else:
            pyautogui.mouseUp(int(cx), int(cy), button='left', _pause=False)

    def click(self, x: Optional[float] = None, y: Optional[float] = None):
        """Performs a single left click."""
        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            down_event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventLeftMouseDown, loc, CG.kCGMouseButtonLeft
            )
            up_event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventLeftMouseUp, loc, CG.kCGMouseButtonLeft
            )
            if down_event and up_event:
                CG.CGEventPost(CG.kCGHIDEventTap, down_event)
                time.sleep(0.01)
                CG.CGEventPost(CG.kCGHIDEventTap, up_event)
        else:
            pyautogui.click(int(cx), int(cy), button='left', _pause=False)

    def double_click(self, x: Optional[float] = None, y: Optional[float] = None):
        """Performs a native macOS double click."""
        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            # Click 1
            d1 = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseDown, loc, CG.kCGMouseButtonLeft)
            u1 = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseUp, loc, CG.kCGMouseButtonLeft)
            CG.CGEventSetIntegerValueField(d1, CG.kCGMouseEventClickState, 1)
            CG.CGEventSetIntegerValueField(u1, CG.kCGMouseEventClickState, 1)
            CG.CGEventPost(CG.kCGHIDEventTap, d1)
            CG.CGEventPost(CG.kCGHIDEventTap, u1)

            time.sleep(0.04)

            # Click 2
            d2 = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseDown, loc, CG.kCGMouseButtonLeft)
            u2 = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseUp, loc, CG.kCGMouseButtonLeft)
            CG.CGEventSetIntegerValueField(d2, CG.kCGMouseEventClickState, 2)
            CG.CGEventSetIntegerValueField(u2, CG.kCGMouseEventClickState, 2)
            CG.CGEventPost(CG.kCGHIDEventTap, d2)
            CG.CGEventPost(CG.kCGHIDEventTap, u2)
        else:
            pyautogui.doubleClick(int(cx), int(cy), _pause=False)

    def right_click(self, x: Optional[float] = None, y: Optional[float] = None):
        """Performs a native macOS right click."""
        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            down_event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventRightMouseDown, loc, CG.kCGMouseButtonRight
            )
            up_event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventRightMouseUp, loc, CG.kCGMouseButtonRight
            )
            if down_event and up_event:
                CG.CGEventPost(CG.kCGHIDEventTap, down_event)
                time.sleep(0.01)
                CG.CGEventPost(CG.kCGHIDEventTap, up_event)
        else:
            pyautogui.click(int(cx), int(cy), button='right', _pause=False)

    def scroll(self, dy: int, dx: int = 0, x: Optional[float] = None, y: Optional[float] = None):
        """
        Scrolls vertically (dy) and horizontally (dx).
        Positive dy = scroll up (content moves down).
        Negative dy = scroll down (content moves up).
        """
        if dy == 0 and dx == 0:
            return

        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            try:
                scroll_event = CG.CGEventCreateScrollWheelEvent2(
                    None,
                    CG.kCGScrollEventUnitLine,
                    2,  # 2 wheel axes (vertical, horizontal)
                    int(dy),
                    int(dx),
                    0
                )
            except Exception:
                scroll_event = CG.CGEventCreateScrollWheelEvent(
                    None,
                    CG.kCGScrollEventUnitLine,
                    1,
                    int(dy)
                )

            if scroll_event:
                CG.CGEventSetLocation(scroll_event, loc)
                CG.CGEventPost(CG.kCGHIDEventTap, scroll_event)
        else:
            pyautogui.scroll(int(dy), x=int(cx), y=int(cy), _pause=False)

    def switch_space_left(self):
        """Move to the Left desktop space (Control + Left Arrow)."""
        threading.Thread(target=_run_system_keystroke, args=(123,), daemon=True).start()

    def switch_space_right(self):
        """Move to the Right desktop space (Control + Right Arrow)."""
        threading.Thread(target=_run_system_keystroke, args=(124,), daemon=True).start()

    def trigger_mission_control(self):
        """Triggers Mission Control directly via macOS launch services."""
        subprocess.Popen(["open", "-a", "Mission Control"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def trigger_shortcut(self, action: str):
        """Dispatches keyboard shortcuts reliably on macOS."""
        scripts = {
            "ENTER": 'tell application "System Events" to key code 36',                         # Return
            "NEW_TAB": 'tell application "System Events" to keystroke "t" using command down',  # Cmd + T
            "CLOSE_TAB": 'tell application "System Events" to keystroke "w" using command down',# Cmd + W
            "ESCAPE": 'tell application "System Events" to key code 53'                          # Escape
        }
        script = scripts.get(action)
        if script:
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



    def release_all(self):
        """Safety release for any held mouse buttons."""
        if self.is_mouse_down:
            self.mouse_up()




