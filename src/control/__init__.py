"""Control package for Open FRIDAY: Mouse simulation, OS permissions, and radial actions."""

from .mouse_controller import MouseController
from .radial_menu import RADIAL_ACTIONS, execute_radial_action
from .permissions import (
    is_accessibility_granted,
    request_accessibility_permission,
    set_macos_accessory_policy,
    setup_macos_fullscreen_overlay,
)

__all__ = [
    "MouseController",
    "RADIAL_ACTIONS",
    "execute_radial_action",
    "is_accessibility_granted",
    "request_accessibility_permission",
    "set_macos_accessory_policy",
    "setup_macos_fullscreen_overlay",
]
