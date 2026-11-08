"""
Customizable Radial Shortcut Menu Actions for Open FRIDAY.
To add, remove, or customize actions, edit RADIAL_ACTIONS below.
The gesture recognizer and UI overlay dynamically compute sector angles,
rendering, and hit testing for any number of actions.
"""

from typing import List, Dict, Any, Optional
import subprocess


RADIAL_ACTIONS: List[Dict[str, Any]] = [
    {
        "id": "NEW_TAB",
        "label": "NEW TAB",
        "symbol": "⌘T",
        "script": 'tell application "System Events" to keystroke "t" using command down',
    },
    {
        "id": "ENTER",
        "label": "ENTER",
        "symbol": "↵",
        "script": 'tell application "System Events" to key code 36',
    },
    {
        "id": "CLOSE_TAB",
        "label": "CLOSE TAB",
        "symbol": "⌘W",
        "script": 'tell application "System Events" to keystroke "w" using command down',
    },
    {
        "id": "ESCAPE",
        "label": "ESCAPE",
        "symbol": "⎋",
        "script": 'tell application "System Events" to key code 53',
    },
]


def execute_radial_action(action_id: str) -> bool:
    """Dispatches the configured shortcut for the given action id."""
    for action in RADIAL_ACTIONS:
        if action["id"] == action_id:
            script = action.get("script")
            if script:
                subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
    return False


def get_radial_action_by_index(index: int) -> Optional[Dict[str, Any]]:
    """Returns the radial action configuration at index modulo len(RADIAL_ACTIONS)."""
    if not RADIAL_ACTIONS:
        return None
    return RADIAL_ACTIONS[index % len(RADIAL_ACTIONS)]

