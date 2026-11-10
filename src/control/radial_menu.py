"""
Customizable Radial Shortcut Menu Actions for Open FRIDAY.
To add, remove, or customize actions, edit RADIAL_ACTIONS below.
The gesture recognizer and UI overlay dynamically compute sector angles,
rendering, and hit testing for any number of actions.
All shortcuts are dispatched natively via macOS Quartz CoreGraphics with zero subprocess overhead.
"""

from typing import List, Dict, Any, Optional

from src.control.keystrokes import post_keyboard_event

try:
    import Quartz.CoreGraphics as CG
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False


RADIAL_ACTIONS: List[Dict[str, Any]] = [
    {
        "id": "NEW_TAB",
        "label": "NEW TAB",
        "symbol": "⌘T",
        "key_code": 17,       # Virtual keycode for 't'
        "flags": "command",
    },
    {
        "id": "ENTER",
        "label": "ENTER",
        "symbol": "↵",
        "key_code": 36,       # Virtual keycode for Return / Enter
        "flags": None,
    },
    {
        "id": "CLOSE_TAB",
        "label": "CLOSE TAB",
        "symbol": "⌘W",
        "key_code": 13,       # Virtual keycode for 'w'
        "flags": "command",
    },
    {
        "id": "ESCAPE",
        "label": "ESCAPE",
        "symbol": "⎋",
        "key_code": 53,       # Virtual keycode for Escape
        "flags": None,
    },
]


def execute_radial_action(action_id: str) -> bool:
    """Dispatches the configured shortcut for the given action id natively via Quartz."""
    for action in RADIAL_ACTIONS:
        if action["id"] == action_id:
            key_code = action.get("key_code")
            if key_code is not None:
                flags = 0
                if QUARTZ_AVAILABLE:
                    flag_type = action.get("flags")
                    if flag_type == "command":
                        flags = CG.kCGEventFlagMaskCommand
                    elif flag_type == "control":
                        flags = CG.kCGEventFlagMaskControl
                    elif flag_type == "shift":
                        flags = CG.kCGEventFlagMaskShift
                    elif flag_type == "option":
                        flags = CG.kCGEventFlagMaskAlternate

                post_keyboard_event(key_code, flags)
                return True
    return False


def get_radial_action_by_index(index: int) -> Optional[Dict[str, Any]]:
    """Returns the radial action configuration at index modulo len(RADIAL_ACTIONS)."""
    if not RADIAL_ACTIONS:
        return None
    return RADIAL_ACTIONS[index % len(RADIAL_ACTIONS)]


