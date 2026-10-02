"""Control package for Open FRIDAY: Mouse simulation, OS accessibility, and desktop controller."""

from .mouse_controller import MouseController
from .computer_controller import MacComputerController
from .accessibility_tree import AccessibilityTreeScraper, UIElementInfo
from .permissions import (
    is_accessibility_granted,
    request_accessibility_permission,
    set_macos_accessory_policy,
)

__all__ = [
    "MouseController",
    "MacComputerController",
    "AccessibilityTreeScraper",
    "UIElementInfo",
    "is_accessibility_granted",
    "request_accessibility_permission",
    "set_macos_accessory_policy",
]
