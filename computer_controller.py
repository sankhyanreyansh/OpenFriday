"""
Native macOS Computer Controller for Open FRIDAY with Set-of-Marks (SoM) Grounding.
Provides high-speed GUI automation using native Quartz CoreGraphics, AppleScript,
macOS Accessibility Tree (AXUIElement) grounding, safe clipboard restoration,
and foveated multi-scale inspection.
"""

import subprocess
import time
import io
import base64
from typing import Tuple, Dict, Any, List, Optional
from PIL import Image, ImageGrab, ImageChops
import Quartz.CoreGraphics as CG
import AppKit

from grid_overlay import draw_visual_coordinate_grid, draw_set_of_marks_overlay
from accessibility_tree import AccessibilityTreeScraper, UIElementInfo


class MacComputerController:
    """Controls mouse, keyboard, and screen perception natively on macOS with Set-of-Marks accuracy."""

    # Virtual Keycode Map for native Quartz keyboard dispatch
    KEY_CODE_MAP: Dict[str, int] = {
        "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
        "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
        "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
        "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38,
        "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46, ".": 47,
        "tab": 48, "space": 49, "`": 50, "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
        "enter": 36, "return": 36, "left": 123, "right": 124, "down": 125, "up": 126,
    }

    def __init__(self):
        main_display = CG.CGMainDisplayID()
        self.logical_width = CG.CGDisplayPixelsWide(main_display)
        self.logical_height = CG.CGDisplayPixelsHigh(main_display)
        self.screen_width = self.logical_width
        self.screen_height = self.logical_height

        self.last_capture_width = self.logical_width
        self.last_capture_height = self.logical_height

        # macOS Accessibility Scraper
        self.scraper = AccessibilityTreeScraper(self.logical_width, self.logical_height)
        self.current_elements_map: Dict[int, UIElementInfo] = {}

    def capture_screen_base64(
        self,
        apply_grid: bool = True,
        apply_som: bool = True
    ) -> Tuple[str, int, int, List[Dict[str, Any]]]:
        """
        Captures primary display into a compressed PNG base64 string.
        Optionally overlays Set-of-Marks (SoM) element badges and 0-1000 coordinate grid.
        Returns:
            (base64_str, target_width, target_height, elements_summary_list)
        """
        screenshot = ImageGrab.grab()
        raw_w, raw_h = screenshot.size

        elements_summary: List[Dict[str, Any]] = []

        if apply_som:
            # Scrape active interactive UI elements
            elements = self.scraper.scrape_interactive_elements(max_elements=40)
            self.current_elements_map = {e.element_id: e for e in elements}

            for e in elements:
                elements_summary.append({
                    "id": e.element_id,
                    "role": e.role_description or e.role,
                    "title": e.title,
                    "value": e.value,
                    "center": e.center,
                    "norm_center": e.norm_center,
                })

            screenshot = draw_set_of_marks_overlay(
                screenshot, elements, self.logical_width, self.logical_height, apply_grid=apply_grid
            )
        elif apply_grid:
            screenshot = draw_visual_coordinate_grid(screenshot)

        # Scale down to max 1280px width while preserving exact aspect ratio
        if raw_w > 1280:
            scale = 1280.0 / float(raw_w)
            target_w = 1280
            target_h = int(round(raw_h * scale))
            resized_img = screenshot.resize((target_w, target_h), Image.Resampling.LANCZOS)
        else:
            target_w = raw_w
            target_h = raw_h
            resized_img = screenshot

        self.last_capture_width = target_w
        self.last_capture_height = target_h

        buf = io.BytesIO()
        resized_img.save(buf, format="PNG", optimize=True)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64_str, target_w, target_h, elements_summary

    def map_coordinates(self, norm_x: int, norm_y: int) -> Tuple[int, int]:
        """Maps normalized (0-1000) coordinates to macOS logical screen points."""
        logical_x = int(round((norm_x / 1000.0) * self.screen_width))
        logical_y = int(round((norm_y / 1000.0) * self.screen_height))
        clamped_x = max(0, min(logical_x, self.screen_width - 1))
        clamped_y = max(0, min(logical_y, self.screen_height - 1))
        return clamped_x, clamped_y

    def click_element(self, element_id: int, button: str = "left", double_click: bool = False) -> str:
        """
        Clicks an interactive UI element with 100% precision using its Set-of-Marks ID.
        """
        elem = self.current_elements_map.get(element_id)
        if not elem:
            return f"Element ID [{element_id}] not found in current Set-of-Marks. Available IDs: {sorted(list(self.current_elements_map.keys()))[:15]}"

        target_x, target_y = elem.center
        print(f"[COMPUTER AGENT DEBUG] SoM Click: Element [{element_id}] '{elem.title}' ({elem.role}) at Logical ({target_x}, {target_y})")

        point = CG.CGPoint(x=target_x, y=target_y)

        # Move cursor to element center
        move_event = CG.CGEventCreateMouseEvent(None, CG.kCGEventMouseMoved, point, CG.kCGMouseButtonLeft)
        CG.CGEventPost(CG.kCGHIDEventTap, move_event)
        time.sleep(0.04)

        btn_lower = button.lower()
        down_type = CG.kCGEventLeftMouseDown if btn_lower == "left" else CG.kCGEventRightMouseDown
        up_type = CG.kCGEventLeftMouseUp if btn_lower == "left" else CG.kCGEventRightMouseUp
        btn = CG.kCGMouseButtonLeft if btn_lower == "left" else CG.kCGMouseButtonRight

        down_event = CG.CGEventCreateMouseEvent(None, down_type, point, btn)
        up_event = CG.CGEventCreateMouseEvent(None, up_type, point, btn)

        CG.CGEventPost(CG.kCGHIDEventTap, down_event)
        time.sleep(0.04)
        CG.CGEventPost(CG.kCGHIDEventTap, up_event)

        if double_click:
            time.sleep(0.08)
            CG.CGEventPost(CG.kCGHIDEventTap, down_event)
            time.sleep(0.04)
            CG.CGEventPost(CG.kCGHIDEventTap, up_event)
            return f"Double-clicked [{element_id}] {elem.role} '{elem.title}' at logical ({target_x}, {target_y})"

        return f"Clicked [{element_id}] {elem.role} '{elem.title}' at logical ({target_x}, {target_y})"

    def click(self, x: int, y: int, button: str = "left", double_click: bool = False, double: bool = False) -> str:
        """
        Direct Quartz click mapped from normalized 0-1000 integer grid to macOS logical points.
        Fallback tool when clicking custom canvas/unlabeled elements.
        """
        is_double = double_click or double
        target_x, target_y = self.map_coordinates(x, y)
        print(f"[COMPUTER AGENT DEBUG] Coordinate Click: Norm({x}, {y}) -> Logical({target_x}, {target_y})")

        point = CG.CGPoint(x=target_x, y=target_y)

        move_event = CG.CGEventCreateMouseEvent(None, CG.kCGEventMouseMoved, point, CG.kCGMouseButtonLeft)
        CG.CGEventPost(CG.kCGHIDEventTap, move_event)
        time.sleep(0.04)

        btn_lower = button.lower()
        down_type = CG.kCGEventLeftMouseDown if btn_lower == "left" else CG.kCGEventRightMouseDown
        up_type = CG.kCGEventLeftMouseUp if btn_lower == "left" else CG.kCGEventRightMouseUp
        btn = CG.kCGMouseButtonLeft if btn_lower == "left" else CG.kCGMouseButtonRight

        down_event = CG.CGEventCreateMouseEvent(None, down_type, point, btn)
        up_event = CG.CGEventCreateMouseEvent(None, up_type, point, btn)

        CG.CGEventPost(CG.kCGHIDEventTap, down_event)
        time.sleep(0.04)
        CG.CGEventPost(CG.kCGHIDEventTap, up_event)

        if is_double:
            time.sleep(0.08)
            CG.CGEventPost(CG.kCGHIDEventTap, down_event)
            time.sleep(0.04)
            CG.CGEventPost(CG.kCGHIDEventTap, up_event)
            return f"Double-clicked {button} at logical ({target_x}, {target_y}) [norm: ({x}, {y})]"

        return f"Clicked {button} at logical ({target_x}, {target_y}) [norm: ({x}, {y})]"

    def hover(self, x: int, y: int) -> str:
        """Moves cursor to normalized (x, y) without clicking to reveal hover tooltips or dropdowns."""
        target_x, target_y = self.map_coordinates(x, y)
        point = CG.CGPoint(x=target_x, y=target_y)
        move_event = CG.CGEventCreateMouseEvent(None, CG.kCGEventMouseMoved, point, CG.kCGMouseButtonLeft)
        CG.CGEventPost(CG.kCGHIDEventTap, move_event)
        time.sleep(0.08)
        return f"Hovered cursor at logical ({target_x}, {target_y}) [norm: ({x}, {y})]"

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 350) -> str:
        """Performs a smooth mouse drag from (start_x, start_y) to (end_x, end_y)."""
        sx, sy = self.map_coordinates(start_x, start_y)
        ex, ey = self.map_coordinates(end_x, end_y)

        start_pt = CG.CGPoint(x=sx, y=sy)
        end_pt = CG.CGPoint(x=ex, y=ey)

        # 1. Move to start and press left mouse down
        CG.CGEventPost(CG.kCGHIDEventTap, CG.CGEventCreateMouseEvent(None, CG.kCGEventMouseMoved, start_pt, CG.kCGMouseButtonLeft))
        time.sleep(0.04)
        CG.CGEventPost(CG.kCGHIDEventTap, CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseDown, start_pt, CG.kCGMouseButtonLeft))
        time.sleep(0.05)

        # 2. Interpolate intermediate drag events
        steps = max(5, int(duration_ms / 30))
        for i in range(1, steps + 1):
            t = float(i) / float(steps)
            cur_x = sx + (ex - sx) * t
            cur_y = sy + (ey - sy) * t
            cur_pt = CG.CGPoint(x=cur_x, y=cur_y)
            drag_event = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseDragged, cur_pt, CG.kCGMouseButtonLeft)
            CG.CGEventPost(CG.kCGHIDEventTap, drag_event)
            time.sleep(0.02)

        # 3. Release mouse at end point
        CG.CGEventPost(CG.kCGHIDEventTap, CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseUp, end_pt, CG.kCGMouseButtonLeft))
        time.sleep(0.04)
        return f"Dragged from logical ({sx}, {sy}) to ({ex}, {ey})"

    def type_text(
        self,
        text: Optional[str] = None,
        content: Optional[str] = None,
        code: Optional[str] = None,
        value: Optional[str] = None,
        string: Optional[str] = None,
        press_enter: bool = False,
        **kwargs
    ) -> str:
        """
        Fast text insertion via clipboard paste with SAFE original clipboard restoration.
        Preserves user's clipboard history completely.
        """
        eff_text = text
        if eff_text is None:
            eff_text = content or code or value or string or kwargs.get("text") or kwargs.get("content") or kwargs.get("code") or ""

        pasteboard = AppKit.NSPasteboard.generalPasteboard()
        saved_types = list(pasteboard.types() or [])
        saved_items = []

        # 1. Backup existing clipboard data
        for p_item in (pasteboard.pasteboardItems() or []):
            item_data = {}
            for t in (p_item.types() or []):
                val = p_item.dataForType_(t)
                if val:
                    item_data[t] = val
            if item_data:
                saved_items.append(item_data)

        # 2. Inject text to clipboard & simulate Cmd+V (if text is non-empty)
        if eff_text:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE, close_fds=True)
            p.communicate(eff_text.encode("utf-8"))

            self.run_applescript('tell application "System Events" to keystroke "v" using command down')
            time.sleep(0.06)

        if press_enter:
            self.run_applescript('tell application "System Events" to key code 36')
            time.sleep(0.04)

        # 3. Restore user's previous clipboard contents cleanly
        if saved_items:
            try:
                pasteboard.clearContents()
                for item_data in saved_items:
                    for t, data in item_data.items():
                        pasteboard.setData_forType_(data, t)
            except Exception as e:
                print(f"[CLIPBOARD RESTORE WARN] Failed to restore pasteboard: {e}")

        display_snippet = eff_text.replace("\n", "\\n")
        return f"Typed text: '{display_snippet[:60]}{'...' if len(display_snippet) > 60 else ''}'"

    def hotkey(self, combo: str) -> str:
        """
        Dispatches native keyboard shortcuts (e.g. 'cmd+t', 'cmd+shift+p', 'cmd+k', 'ctrl+c', 'option+tab').
        """
        parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
        if not parts:
            return "Empty hotkey"

        key_name = parts[-1]
        modifiers = parts[:-1]

        mod_script_parts = []
        for m in modifiers:
            if m in ("cmd", "command"):
                mod_script_parts.append("command down")
            elif m in ("shift", "sh"):
                mod_script_parts.append("shift down")
            elif m in ("ctrl", "control"):
                mod_script_parts.append("control down")
            elif m in ("alt", "opt", "option"):
                mod_script_parts.append("option down")

        mod_str = f" using {{{', '.join(mod_script_parts)}}}" if mod_script_parts else ""

        key_code = self.KEY_CODE_MAP.get(key_name)
        if key_code is not None:
            script = f'tell application "System Events" to key code {key_code}{mod_str}'
        else:
            script = f'tell application "System Events" to keystroke "{key_name}"{mod_str}'

        self.run_applescript(script)
        time.sleep(0.05)
        return f"Dispatched hotkey: {combo}"

    def key_press(self, key_name: str, modifiers: Optional[List[str]] = None) -> str:
        """Dispatches keyboard press with optional modifiers."""
        if modifiers:
            combo = "+".join(modifiers + [key_name])
            return self.hotkey(combo)
        return self.hotkey(key_name)

    def scroll(self, dy: int = 0, dx: int = 0) -> str:
        """Scrolls vertically (dy) or horizontally (dx) via Quartz."""
        scroll_event = CG.CGEventCreateScrollWheelEvent2(
            None,
            CG.kCGScrollEventUnitPixel,
            2,
            int(dy),
            int(dx),
            0
        )
        CG.CGEventPost(CG.kCGHIDEventTap, scroll_event)
        time.sleep(0.05)
        return f"Scrolled dy={dy}, dx={dx}"

    def inspect_region(self, x: int, y: int, radius: int = 150) -> Tuple[str, int, int]:
        """
        Foveated High-Resolution Perception:
        Crops a 1:1 uncompressed pixel crop around (x, y) normalized coordinate
        and returns base64 string for fine-grained reading of dense text, code, or small buttons.
        """
        screenshot = ImageGrab.grab()
        img_w, img_h = screenshot.size

        lx, ly = self.map_coordinates(x, y)
        scale_x = float(img_w) / float(max(1, self.logical_width))
        scale_y = float(img_h) / float(max(1, self.logical_height))

        px = int(round(lx * scale_x))
        py = int(round(ly * scale_y))
        pr = int(round(radius * scale_x))

        x1 = max(0, px - pr)
        y1 = max(0, py - pr)
        x2 = min(img_w, px + pr)
        y2 = min(img_h, py + pr)

        cropped = screenshot.crop((x1, y1, x2, y2))
        cw, ch = cropped.size

        buf = io.BytesIO()
        cropped.save(buf, format="PNG", optimize=True)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64_str, cw, ch

    def wait_for_change(self, timeout_ms: int = 1200) -> str:
        """
        Perceptual Settling Check:
        Waits until sequential screenshot frames stop changing rapidly (e.g. after page load).
        """
        start_t = time.time()
        initial = ImageGrab.grab()
        prev = initial

        while (time.time() - start_t) * 1000 < timeout_ms:
            time.sleep(0.12)
            curr = ImageGrab.grab()
            diff = ImageChops.difference(prev, curr)
            bbox = diff.getbbox()
            if not bbox:
                return "Screen state settled."
            prev = curr

        return "Wait completed."

    def run_applescript(self, script: str) -> str:
        """Executes AppleScript command via osascript."""
        try:
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
            return res.stdout.strip()
        except Exception as e:
            return f"AppleScript error: {e}"
