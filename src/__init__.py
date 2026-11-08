"""
Open FRIDAY Source Package.
Organized into domain-specific subpackages:
- src.vision: Camera capture, hand landmark tracking, and gesture recognition
- src.control: Mouse simulation, OS permissions, and radial shortcut actions
- src.ui: Transparent reticle overlay and glassmorphic HUD status pill
"""

from . import vision
from . import control
from . import ui

__all__ = [
    "vision",
    "control",
    "ui",
]
