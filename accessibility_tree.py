"""
macOS Native Accessibility Tree (AXUIElement) Scraper for Open FRIDAY.
Extracts interactive UI elements (buttons, text fields, menu items, links, tabs)
from the frontmost macOS application and active window hierarchy for Set-of-Marks (SoM) grounding.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Set
import AppKit

try:
    import ApplicationServices as AX
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False


@dataclass
class UIElementInfo:
    """Represents an interactive macOS UI element extracted via AXUIElement."""
    element_id: int
    role: str
    role_description: str
    title: str
    value: str
    bounds: Tuple[int, int, int, int]  # (x, y, width, height) in logical points
    center: Tuple[int, int]            # (center_x, center_y) in logical points
    norm_center: Tuple[int, int]       # (0-1000 norm_x, 0-1000 norm_y)
    norm_bounds: Tuple[int, int, int, int]  # (norm_x, norm_y, norm_w, norm_h)


class AccessibilityTreeScraper:
    """Scrapes interactive UI hierarchy from the active macOS application using ApplicationServices."""

    INTERACTIVE_ROLES: Set[str] = {
        "AXButton",
        "AXTextField",
        "AXTextArea",
        "AXMenuItem",
        "AXMenuButton",
        "AXLink",
        "AXPopUpButton",
        "AXCheckBox",
        "AXRadioButton",
        "AXCell",
        "AXTab",
        "AXComboBox",
        "AXSlider",
        "AXIncrementer",
        "AXDisclosureTriangle",
        "AXImage",
        "AXHeading",
    }

    CONTAINER_ROLES: Set[str] = {
        "AXApplication",
        "AXWindow",
        "AXGroup",
        "AXScrollArea",
        "AXWebArea",
        "AXSplitGroup",
        "AXList",
        "AXTable",
        "AXRow",
        "AXColumn",
        "AXLayoutArea",
        "AXLayoutItem",
        "AXToolbar",
        "AXTabGroup",
    }

    def __init__(self, screen_width: int = 1470, screen_height: int = 956):
        self.screen_width = max(1, screen_width)
        self.screen_height = max(1, screen_height)
        self.ax_available = AX_AVAILABLE

    def update_screen_size(self, width: int, height: int):
        self.screen_width = max(1, width)
        self.screen_height = max(1, height)

    def get_frontmost_app_info(self) -> Optional[Tuple[str, int]]:
        """Returns (app_name, pid) for the frontmost application."""
        try:
            ws = AppKit.NSWorkspace.sharedWorkspace()
            front_app = ws.frontmostApplication()
            if front_app:
                name = str(front_app.localizedName() or "Unknown")
                pid = int(front_app.processIdentifier())
                return name, pid
        except Exception as e:
            print(f"[ACCESSIBILITY TREE ERROR] Failed to get frontmost app: {e}")
        return None

    def _get_attribute(self, element, attribute_name: str) -> Any:
        """Helper to copy an attribute value from an AXUIElement."""
        if not self.ax_available or element is None:
            return None
        try:
            err, val = AX.AXUIElementCopyAttributeValue(element, attribute_name, None)
            return val if err == 0 else None
        except Exception:
            return None

    def _extract_bounds(self, element) -> Optional[Tuple[int, int, int, int]]:
        """Extracts (x, y, w, h) logical bounds from an AXUIElement."""
        if not self.ax_available or element is None:
            return None

        pos_val = self._get_attribute(element, AX.kAXPositionAttribute)
        size_val = self._get_attribute(element, AX.kAXSizeAttribute)

        if pos_val is None or size_val is None:
            return None

        try:
            ok_p, pt = AX.AXValueGetValue(pos_val, AX.kAXValueCGPointType, None)
            ok_s, sz = AX.AXValueGetValue(size_val, AX.kAXValueCGSizeType, None)

            if ok_p and ok_s:
                x = int(round(pt.x))
                y = int(round(pt.y))
                w = int(round(sz.width))
                h = int(round(sz.height))
                return x, y, w, h
        except Exception:
            pass

        return None

    def scrape_interactive_elements(
        self,
        max_elements: int = 40,
        max_depth: int = 9
    ) -> List[UIElementInfo]:
        """
        Traverses the frontmost application's accessibility tree and returns a list
        of visible, interactive UI elements with Set-of-Marks IDs and coordinates.
        """
        if not self.ax_available:
            return []

        app_info = self.get_frontmost_app_info()
        if not app_info:
            return []

        app_name, pid = app_info
        app_elem = AX.AXUIElementCreateApplication(pid)
        if not app_elem:
            return []

        # Turn on enhanced UI / accessibility for Chrome/Electron applications
        try:
            AX.AXUIElementSetAttributeValue(app_elem, "AXEnhancedUserInterface", True)
            AX.AXUIElementSetAttributeValue(app_elem, "AXManualAccessibility", True)
        except Exception:
            pass

        elements: List[UIElementInfo] = []
        element_counter = 1
        seen_centers: Set[Tuple[int, int]] = set()

        def traverse(node, depth: int):
            nonlocal element_counter
            if not node or depth > max_depth or len(elements) >= max_elements:
                return

            role = str(self._get_attribute(node, AX.kAXRoleAttribute) or "")
            role_desc = str(self._get_attribute(node, AX.kAXRoleDescriptionAttribute) or "")
            title = str(self._get_attribute(node, AX.kAXTitleAttribute) or "")
            val = self._get_attribute(node, AX.kAXValueAttribute)
            val_str = str(val) if (val is not None and isinstance(val, (str, int, float))) else ""
            desc = str(self._get_attribute(node, AX.kAXDescriptionAttribute) or "")
            label = title or desc or val_str

            is_interactive = (role in self.INTERACTIVE_ROLES) or (
                bool(label) and role not in self.CONTAINER_ROLES
            )

            if is_interactive:
                bounds = self._extract_bounds(node)
                if bounds:
                    x, y, w, h = bounds
                    # Filter out zero-size, tiny, or off-screen elements
                    if (
                        w >= 10 and h >= 10 and
                        w < self.screen_width * 0.98 and
                        h < self.screen_height * 0.98 and
                        x + w > 0 and x < self.screen_width and
                        y + h > 0 and y < self.screen_height
                    ):
                        cx = x + (w // 2)
                        cy = y + (h // 2)

                        # Deduplicate elements occupying nearly the same center (within 10px)
                        center_key = (cx // 10, cy // 10)
                        if center_key not in seen_centers:
                            seen_centers.add(center_key)

                            norm_cx = int(round((cx / float(self.screen_width)) * 1000.0))
                            norm_cy = int(round((cy / float(self.screen_height)) * 1000.0))
                            norm_x = int(round((x / float(self.screen_width)) * 1000.0))
                            norm_y = int(round((y / float(self.screen_height)) * 1000.0))
                            norm_w = int(round((w / float(self.screen_width)) * 1000.0))
                            norm_h = int(round((h / float(self.screen_height)) * 1000.0))

                            clean_label = label[:60] if label else role
                            clean_val = val_str[:40] if (val_str and val_str != clean_label) else ""

                            info = UIElementInfo(
                                element_id=element_counter,
                                role=role,
                                role_description=role_desc or role,
                                title=clean_label,
                                value=clean_val,
                                bounds=(x, y, w, h),
                                center=(cx, cy),
                                norm_center=(norm_cx, norm_cy),
                                norm_bounds=(norm_x, norm_y, norm_w, norm_h),
                            )
                            elements.append(info)
                            element_counter += 1

            # Traverse children
            children = self._get_attribute(node, AX.kAXChildrenAttribute)
            if children:
                for child in children:
                    if len(elements) >= max_elements:
                        break
                    traverse(child, depth + 1)

        windows = self._get_attribute(app_elem, AX.kAXWindowsAttribute)
        for win in (windows or []):
            traverse(win, depth=0)
            if len(elements) >= max_elements:
                break

        return elements
