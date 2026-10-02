"""
Open FRIDAY Source Package.
Organized into domain-specific subpackages:
- src.vision: Camera capture, hand landmark tracking, and gesture recognition
- src.control: Mouse simulation, desktop automation, accessibility tree, and permissions
- src.audio: Voice dictation and Whisper transcription
- src.ui: Transparent reticle, AR screen annotations, HUD sidebar, and coordinate grid
- src.ai_assistant: Conversational intelligence, autonomous computer agent, tools, memory vault, and bash executor
"""

import sys
import os

# Auto-configure search paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

for sub in ["", "vision", "control", "audio", "ui", "ai_assistant"]:
    p = os.path.join(SRC_DIR, sub) if sub else SRC_DIR
    if p not in sys.path:
        sys.path.insert(0, p)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Expose subpackages
from . import vision
from . import control
from . import audio
from . import ui
from . import ai_assistant

# Alias modules into sys.modules to prevent duplicate module loads and Enum identity desync
from src.vision import gesture_recognizer, vision_engine, landmark_smoother, one_euro_filter
from src.control import mouse_controller, computer_controller, accessibility_tree, permissions
from src.audio import dictation_engine
from src.ui import hud_sidebar, overlay_window, annotation_overlay, grid_overlay
from src.ai_assistant import assistant as ai_assistant_mod, memory_vault, bash_executor, system_tools

_compat_aliases = {
    "gesture_recognizer": gesture_recognizer,
    "vision_engine": vision_engine,
    "landmark_smoother": landmark_smoother,
    "one_euro_filter": one_euro_filter,
    "mouse_controller": mouse_controller,
    "computer_controller": computer_controller,
    "accessibility_tree": accessibility_tree,
    "permissions": permissions,
    "dictation_engine": dictation_engine,
    "hud_sidebar": hud_sidebar,
    "overlay_window": overlay_window,
    "annotation_overlay": annotation_overlay,
    "grid_overlay": grid_overlay,
    "ai_assistant": ai_assistant,
    "memory_vault": memory_vault,
    "bash_executor": bash_executor,
    "system_tools": system_tools,
}

for name, mod in _compat_aliases.items():
    sys.modules[name] = mod

__all__ = [
    "vision",
    "control",
    "audio",
    "ui",
    "ai_assistant",
]
