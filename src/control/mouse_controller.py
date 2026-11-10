"""
macOS Native Mouse & Cursor Controller using Quartz CoreGraphics.
Provides low-latency, high-precision cursor movement, left/right clicks,
drag-and-drop, double clicks, linear bidirectional scrolling, space switching,
zoom, and radial shortcut execution directly into the macOS Window Server.
"""

import time
import subprocess
import threading
import queue
import logging
from typing import Tuple, Optional

from src.control.keystrokes import post_keyboard_event, KEY_PLUS, KEY_MINUS
from src.control.radial_menu import execute_radial_action

try:
    import Quartz.CoreGraphics as CG
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0

logger = logging.getLogger("OpenFriday.MouseController")

# Dedicated FIFO queue & worker for macOS Spaces & Mission Control events
# Guarantees sequential execution so rapid successive flicks never race or complete out of order
_system_keystroke_queue: "queue.Queue[int]" = queue.Queue()


def _system_keystroke_worker():
    """
    Dedicated FIFO background worker for system keystrokes (Spaces / Mission Control).
    Processes keystrokes sequentially so rapid successive flicks never race or execute out of order.
    """
    while True:
        key_code = _system_keystroke_queue.get()
        try:
            # macOS intentionally filters out synthetic CGEvents for Mission Control / Spaces switching;
            # AppleScript via System Events is the officially sanctioned and working mechanism.
            script = f'tell application "System Events" to key code {key_code} using control down'
            subprocess.run(
                ["osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0
            )
        except Exception as e:
            logger.warning("Error running system keystroke %s: %s", key_code, e)
        finally:
            _system_keystroke_queue.task_done()


_keystroke_thread = threading.Thread(target=_system_keystroke_worker, daemon=True, name="SystemKeystrokeWorker")
_keystroke_thread.start()


def _run_system_keystroke(key_code: int):
    """
    Enqueues Control + KeyCode for reliable, ordered dispatch to macOS System Events.
    Non-blocking O(1) enqueue on the calling thread.
    """
    _system_keystroke_queue.put(key_code)


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
        """Performs a native left click with distinct timestamped down and up events."""
        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            now_ns = time.monotonic_ns()
            down_event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventLeftMouseDown, loc, CG.kCGMouseButtonLeft
            )
            up_event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventLeftMouseUp, loc, CG.kCGMouseButtonLeft
            )
            if down_event and up_event:
                CG.CGEventSetTimestamp(down_event, now_ns)
                CG.CGEventSetTimestamp(up_event, now_ns + 10_000_000)
                CG.CGEventSetIntegerValueField(down_event, CG.kCGMouseEventClickState, 1)
                CG.CGEventSetIntegerValueField(up_event, CG.kCGMouseEventClickState, 1)
                CG.CGEventPost(CG.kCGHIDEventTap, down_event)
                CG.CGEventPost(CG.kCGHIDEventTap, up_event)
        else:
            pyautogui.click(int(cx), int(cy), button='left', _pause=False)

    def double_click(self, x: Optional[float] = None, y: Optional[float] = None):
        """Performs a native macOS double click with distinct timestamps and clickState 1 and 2."""
        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            now_ns = time.monotonic_ns()
            d1 = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseDown, loc, CG.kCGMouseButtonLeft)
            u1 = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseUp, loc, CG.kCGMouseButtonLeft)
            d2 = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseDown, loc, CG.kCGMouseButtonLeft)
            u2 = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseUp, loc, CG.kCGMouseButtonLeft)

            CG.CGEventSetTimestamp(d1, now_ns)
            CG.CGEventSetTimestamp(u1, now_ns + 15_000_000)
            CG.CGEventSetTimestamp(d2, now_ns + 30_000_000)
            CG.CGEventSetTimestamp(u2, now_ns + 45_000_000)

            CG.CGEventSetIntegerValueField(d1, CG.kCGMouseEventClickState, 1)
            CG.CGEventSetIntegerValueField(u1, CG.kCGMouseEventClickState, 1)
            CG.CGEventSetIntegerValueField(d2, CG.kCGMouseEventClickState, 2)
            CG.CGEventSetIntegerValueField(u2, CG.kCGMouseEventClickState, 2)

            CG.CGEventPost(CG.kCGHIDEventTap, d1)
            CG.CGEventPost(CG.kCGHIDEventTap, u1)
            CG.CGEventPost(CG.kCGHIDEventTap, d2)
            CG.CGEventPost(CG.kCGHIDEventTap, u2)
        else:
            pyautogui.doubleClick(int(cx), int(cy), _pause=False)

    def right_click(self, x: Optional[float] = None, y: Optional[float] = None):
        """Performs a native macOS right click without blocking sleeps."""
        if x is not None and y is not None:
            cx, cy = self.clamp(x, y)
            self.current_x = cx
            self.current_y = cy
        else:
            cx, cy = self.current_x, self.current_y

        if self.quartz_available:
            loc = CG.CGPoint(cx, cy)
            now_ns = time.monotonic_ns()
            down_event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventRightMouseDown, loc, CG.kCGMouseButtonRight
            )
            up_event = CG.CGEventCreateMouseEvent(
                None, CG.kCGEventRightMouseUp, loc, CG.kCGMouseButtonRight
            )
            if down_event and up_event:
                CG.CGEventSetTimestamp(down_event, now_ns)
                CG.CGEventSetTimestamp(up_event, now_ns + 10_000_000)
                CG.CGEventSetIntegerValueField(down_event, CG.kCGMouseEventClickState, 1)
                CG.CGEventSetIntegerValueField(up_event, CG.kCGMouseEventClickState, 1)
                CG.CGEventPost(CG.kCGHIDEventTap, down_event)
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
                    2,
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
        """Move to the Left desktop space (Control + Left Arrow) via macOS System Events."""
        _run_system_keystroke(123)

    def switch_space_right(self):
        """Move to the Right desktop space (Control + Right Arrow) via macOS System Events."""
        _run_system_keystroke(124)

    def trigger_mission_control(self):
        """Triggers Mission Control (Control + Up Arrow) via macOS System Events."""
        _run_system_keystroke(126)

    def trigger_app_expose(self):
        """Triggers Application Windows / App Exposé (Control + Down Arrow) via macOS System Events."""
        _run_system_keystroke(125)

    # macOS Virtual Keycodes for Zoom
    KEY_PLUS = KEY_PLUS
    KEY_MINUS = KEY_MINUS

    def zoom(self, direction: str, steps: int = 1):
        """
        Simulates macOS Command + Plus (Zoom In) or Command + Minus (Zoom Out) natively.
        direction: 'in' or 'out'
        steps: number of zoom keystrokes to fire
        """
        keycode = self.KEY_PLUS if direction.lower() == "in" else self.KEY_MINUS
        cmd_flag = CG.kCGEventFlagMaskCommand if self.quartz_available else 0

        for _ in range(steps):
            post_keyboard_event(keycode, cmd_flag)

    def trigger_shortcut(self, action_id: str):
        """Dispatches keyboard shortcuts defined in the radial menu system."""
        execute_radial_action(action_id)

    def release_all(self):
        """Safety release for any held mouse buttons."""
        if self.is_mouse_down:
            self.mouse_up()
