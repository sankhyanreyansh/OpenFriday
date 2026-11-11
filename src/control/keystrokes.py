"""
macOS Native Keyboard Event Dispatcher using Quartz CoreGraphics.
Dispatches synthetic key down and key up events directly to the macOS Window Server
with sub-millisecond latency and zero subprocess overhead.
"""

import time
import logging

try:
    import Quartz.CoreGraphics as CG
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False

logger = logging.getLogger("OpenFriday.Keystrokes")

# Common Virtual Keycodes for macOS
KEY_C = 8             # Key 'c' on ANSI keyboard (Cmd+C copy)
KEY_V = 9             # Key 'v' on ANSI keyboard (Cmd+V paste)
KEY_W = 13            # Key 'w' on ANSI keyboard (Cmd+W close tab)
KEY_T = 17            # Key 't' on ANSI keyboard (Cmd+T new tab)
KEY_PLUS = 0x18       # Key '=' / '+' on ANSI keyboard
KEY_MINUS = 0x1B      # Key '-' on ANSI keyboard
KEY_RETURN = 36       # Key 'Return' / Enter on ANSI keyboard
KEY_DELETE = 51       # Key 'Delete' / Backspace on ANSI keyboard
KEY_ESCAPE = 53       # Key 'Escape' on ANSI keyboard
KEY_LEFT_ARROW = 123
KEY_RIGHT_ARROW = 124
KEY_DOWN_ARROW = 125
KEY_UP_ARROW = 126


def post_keyboard_event(key_code: int, flags: int = 0):
    """
    Dispatches a native key down + key up event directly into the macOS Window Server.
    Uses time.monotonic_ns() for boot-relative event timestamps required by CoreGraphics.
    """
    if QUARTZ_AVAILABLE:
        now_ns = time.monotonic_ns()
        down_event = CG.CGEventCreateKeyboardEvent(None, key_code, True)
        if down_event:
            CG.CGEventSetTimestamp(down_event, now_ns)
            if flags:
                CG.CGEventSetFlags(down_event, flags)
            CG.CGEventPost(CG.kCGHIDEventTap, down_event)

        up_event = CG.CGEventCreateKeyboardEvent(None, key_code, False)
        if up_event:
            # Post release 15ms after down event
            CG.CGEventSetTimestamp(up_event, now_ns + 15_000_000)
            if flags:
                CG.CGEventSetFlags(up_event, flags)
            CG.CGEventPost(CG.kCGHIDEventTap, up_event)
    else:
        logger.debug("Quartz unavailable; post_keyboard_event(%s, flags=%s) skipped", key_code, flags)

