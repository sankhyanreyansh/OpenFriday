"""
Dual-Hand Modifier Gesture Recognition Engine for Open FRIDAY.
- Dominant Hand (Right Hand): Cursor navigation (Index MCP), pinch click, double click, dragging, scrolling.
- Modifier Hand (Left Hand): Physical modifier (Shift key) when raised in frame (converts Left Click to Right Click).
- Dual Hand Gestures:
    - Dual Open Palms (Hold 250ms): Summons customizable Radial Shortcut Menu.
    - Dual Pinch: Pinch-to-Zoom (zoom in / zoom out).
- Three-finger Swipes:
    - Swipe Left: Next Desktop Space
    - Swipe Right: Previous Desktop Space
    - Swipe Up: Mission Control
    - Swipe Down: App Exposé / Application Windows
- Two-finger Scroll: Inertial momentum decay scrolling.
"""

import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

from src.control.radial_menu import RADIAL_ACTIONS


class GestureState(Enum):
    NONE = "NONE"
    POINTING = "POINTING"
    PINCHING = "PINCHING"
    CLICK = "CLICK"
    DRAGGING = "DRAGGING"
    DOUBLE_CLICK = "DOUBLE CLICK"
    RIGHT_CLICK = "RIGHT CLICK"
    SCROLLING = "SCROLLING"
    SWIPE_NAV = "SWIPE NAV"
    RADIAL_MENU = "RADIAL MENU"
    ZOOM = "ZOOM"


@dataclass
class GestureData:
    """Carries full state information of the detected gesture per frame."""
    raw_x: float = 0.0
    raw_y: float = 0.0
    screen_x: float = 0.0
    screen_y: float = 0.0
    state: GestureState = GestureState.NONE
    pinch_progress: float = 0.0
    scroll_dy: int = 0
    hand_landmarks: Optional[List[Any]] = None
    is_tracking: bool = False
    confidence: float = 0.0
    fps: float = 0.0
    status_message: str = "Ready"
    is_modifier_active: bool = False
    nav_action: Optional[str] = None
    radial_sector: Optional[str] = None
    drag_start_x: float = 0.0
    drag_start_y: float = 0.0


class GestureRecognizer:
    """Analyzes 3D MediaPipe landmarks for high-accuracy desktop control."""

    # Landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self):
        # Pinch contact thresholds (normalized to palm scale)
        self.pinch_start_threshold = 0.38
        self.pinch_release_threshold = 0.52

        # Timing and travel thresholds
        self.tap_max_time = 0.35          # Tap duration < 350ms
        self.drag_dist_threshold = 38.0   # Spatial travel threshold for drag (38px isolates clicks from drags)
        self.double_click_window = 0.55   # Double-tap window (550ms for natural air double-taps)
        self.double_click_dist = 60.0     # Spatial tolerance between two air taps (60px)

        # State tracking
        self.is_pinching = False
        self.is_dragging = False
        self.pinch_start_time: float = 0.0
        self.pinch_freeze_x: float = 0.0
        self.pinch_freeze_y: float = 0.0
        self.pinch_start_modifier: bool = False

        self.last_tap_time: float = 0.0
        self.last_tap_x: float = 0.0
        self.last_tap_y: float = 0.0

        # Two-finger scrolling & momentum
        self.prev_scroll_y: Optional[float] = None
        self.scroll_accumulator: float = 0.0
        self.scroll_sensitivity = 1.0
        self.scroll_velocity_y: float = 0.0
        self.is_momentum_scrolling: bool = False
        self.last_scroll_time: Optional[float] = None
        self.last_momentum_time: Optional[float] = None
        self.MOMENTUM_MIN_VELOCITY: float = 3.0
        self.MOMENTUM_DECAY_RATE: float = 6.5

        # Three-finger space navigation with turning-point inflection detection
        self.three_finger_history: List[Tuple[float, float, float]] = []
        self.last_swipe_time: float = 0.0
        self.last_swipe_dir: Optional[str] = None
        self.SAME_DIR_COOLDOWN: float = 0.16       # 160ms for fast repeated flicks in same direction
        self.OPPOSITE_DIR_COOLDOWN: float = 0.38   # 380ms recoil guard against bounce-back triggers
        self.PERP_DIR_COOLDOWN: float = 0.20       # 200ms between horizontal and vertical swipes
        self.SWIPE_DIST_THRESHOLD: float = 0.052   # 5.2% of camera frame width
        self.MIN_SWIPE_SPEED: float = 0.38         # Normalized velocity threshold
        self.is_in_swipe_mode: bool = False
        self.three_finger_detect_frames: int = 0
        self.three_finger_lost_frames: int = 0

        # Radial menu state
        self.dual_open_start_time: Optional[float] = None
        self.is_radial_active: bool = False
        self.DUAL_OPEN_HOLD_THRESHOLD: float = 0.25
        self.screen_center: Tuple[float, float] = (1470.0 / 2.0, 956.0 / 2.0)

        # Dual-hand pinch zoom tracking
        self.dual_pinch_active: bool = False
        self.prev_dual_pinch_dist: Optional[float] = None
        self.zoom_deadzone: float = 0.015

    def set_screen_dimensions(self, width: float, height: float):
        """Sets the screen center coordinates for radial wheel calculations."""
        self.screen_center = (width / 2.0, height / 2.0)

    def dismiss_radial_menu(self):
        """Manually dismisses the radial menu modal state."""
        self.is_radial_active = False
        self.dual_open_start_time = None

    # -------------------------------------------------------------------------
    # Geometric and Finger State Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_lm_xy(lm) -> Tuple[float, float]:
        if hasattr(lm, 'x'):
            return lm.x, lm.y
        return lm[0], lm[1]

    def _euclidean_dist(self, p1, p2) -> float:
        if hasattr(p1, 'x') and hasattr(p2, 'x'):
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            dz = getattr(p1, 'z', 0.0) - getattr(p2, 'z', 0.0)
        else:
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dz = (p1[2] - p2[2]) if len(p1) > 2 and len(p2) > 2 else 0.0
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _get_palm_scale(self, landmarks) -> float:
        """Computes reference scale based on wrist to middle MCP and palm breadth."""
        wrist = landmarks[self.WRIST]
        middle_mcp = landmarks[self.MIDDLE_MCP]
        index_mcp = landmarks[self.INDEX_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]

        wrist_to_middle = self._euclidean_dist(wrist, middle_mcp)
        palm_breadth = self._euclidean_dist(index_mcp, pinky_mcp)
        return max((wrist_to_middle + palm_breadth) * 0.5, 0.01)

    def _is_finger_extended(self, landmarks, tip_idx: int, pip_idx: int, mcp_idx: Optional[int] = None) -> bool:
        """Tilt- and rotation-invariant 3D extension detection using MCP-to-TIP phalanx ratio."""
        if landmarks is None or len(landmarks) < 21:
            return False
        if mcp_idx is not None:
            d_tip_mcp = self._euclidean_dist(landmarks[tip_idx], landmarks[mcp_idx])
            d_pip_mcp = self._euclidean_dist(landmarks[pip_idx], landmarks[mcp_idx])
            if d_pip_mcp > 0.001:
                return (d_tip_mcp / d_pip_mcp) > 1.40
        wrist = landmarks[self.WRIST]
        d_tip_wrist = self._euclidean_dist(landmarks[tip_idx], wrist)
        d_pip_wrist = self._euclidean_dist(landmarks[pip_idx], wrist)
        return d_tip_wrist > d_pip_wrist * 1.08

    def _is_finger_curled(self, landmarks, tip_idx: int, pip_idx: int, mcp_idx: Optional[int] = None) -> bool:
        """Tilt- and rotation-invariant 3D curled detection."""
        if landmarks is None or len(landmarks) < 21:
            return False
        if mcp_idx is not None:
            d_tip_mcp = self._euclidean_dist(landmarks[tip_idx], landmarks[mcp_idx])
            d_pip_mcp = self._euclidean_dist(landmarks[pip_idx], landmarks[mcp_idx])
            if d_pip_mcp > 0.001:
                return (d_tip_mcp / d_pip_mcp) < 1.25
        wrist = landmarks[self.WRIST]
        d_tip_wrist = self._euclidean_dist(landmarks[tip_idx], wrist)
        d_pip_wrist = self._euclidean_dist(landmarks[pip_idx], wrist)
        return d_tip_wrist < d_pip_wrist * 0.98

    def _is_thumb_extended(self, landmarks) -> bool:
        if landmarks is None or len(landmarks) < 21:
            return False
        wrist = landmarks[self.WRIST]
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_mcp = landmarks[self.THUMB_MCP]
        index_mcp = landmarks[self.INDEX_MCP]
        return (
            self._euclidean_dist(thumb_tip, wrist) > self._euclidean_dist(thumb_mcp, wrist) * 1.08 and
            self._euclidean_dist(thumb_tip, index_mcp) > self._euclidean_dist(thumb_mcp, index_mcp) * 0.85
        )

    def _is_hand_fully_open(self, landmarks) -> bool:
        if landmarks is None or len(landmarks) < 21:
            return False
        return (
            self._is_thumb_extended(landmarks) and
            self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP) and
            self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP) and
            self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP) and
            self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)
        )

    def is_scrolling_posture(self, landmarks) -> bool:
        """
        Detects if hand is in two-finger scrolling posture:
        Index and Middle extended, Middle not curled into palm, and NOT in pinch contact.
        """
        if landmarks is None or len(landmarks) < 21:
            return False

        if self.is_hand_pinched(landmarks):
            return False

        idx_ext = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        mid_ext = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        mid_curled = self._is_finger_curled(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)

        return (idx_ext or mid_ext) and (not mid_curled)

    def is_hand_pinched(self, landmarks) -> bool:
        """
        Evaluates if hand configuration is in an active pinch contact.
        Watertight rules:
        1. In two-finger scroll, index tip is closer to middle tip than thumb. In pinch, index touches thumb!
        2. Index-to-thumb distance normalized to palm scale < pinch_start_threshold.
        """
        if landmarks is None or len(landmarks) < 21:
            return False

        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]

        d_index_thumb = self._euclidean_dist(index_tip, thumb_tip)
        d_index_middle = self._euclidean_dist(index_tip, middle_tip)

        # In two-finger scroll, index and middle tips are side-by-side, closer than index-to-thumb
        if d_index_middle < d_index_thumb:
            return False

        palm_scale = self._get_palm_scale(landmarks)
        d_tip = d_index_thumb / palm_scale
        return d_tip < self.pinch_start_threshold

    def get_stable_pointer_coords(self, landmarks) -> Tuple[float, float]:
        """Drives cursor navigation directly from Index MCP (Landmark 5, base knuckle)."""
        return self._get_lm_xy(landmarks[self.INDEX_MCP])

    def cancel_momentum_scroll(self):
        """Immediately cancels ongoing inertial momentum scrolling."""
        self.is_momentum_scrolling = False
        self.scroll_velocity_y = 0.0
        self.scroll_accumulator = 0.0
        self.last_momentum_time = None

    # -------------------------------------------------------------------------
    # Dual-Hand Zoom
    # -------------------------------------------------------------------------

    def process_dual_hand_zoom(
        self,
        left_landmarks,
        right_landmarks,
        left_pinched: bool,
        right_pinched: bool,
    ) -> float:
        """Returns zoom delta: >0 for Zoom In (hands apart), <0 for Zoom Out (hands closer)."""
        if not (left_pinched and right_pinched and left_landmarks and right_landmarks):
            self.dual_pinch_active = False
            self.prev_dual_pinch_dist = None
            return 0.0

        lx1, ly1 = self._get_lm_xy(left_landmarks[self.THUMB_TIP])
        lx2, ly2 = self._get_lm_xy(left_landmarks[self.INDEX_TIP])
        lx = (lx1 + lx2) / 2.0
        ly = (ly1 + ly2) / 2.0

        rx1, ry1 = self._get_lm_xy(right_landmarks[self.THUMB_TIP])
        rx2, ry2 = self._get_lm_xy(right_landmarks[self.INDEX_TIP])
        rx = (rx1 + rx2) / 2.0
        ry = (ry1 + ry2) / 2.0

        current_dist = math.hypot(rx - lx, ry - ly)

        if not self.dual_pinch_active or self.prev_dual_pinch_dist is None:
            self.dual_pinch_active = True
            self.prev_dual_pinch_dist = current_dist
            return 0.0

        dist_delta = current_dist - self.prev_dual_pinch_dist
        if abs(dist_delta) < self.zoom_deadzone:
            return 0.0

        self.prev_dual_pinch_dist = current_dist
        return dist_delta

    # -------------------------------------------------------------------------
    # Sub-Evaluators for Primary Gesture Loop
    # -------------------------------------------------------------------------

    def _eval_active_radial_menu(
        self,
        landmarks: Optional[List[Any]],
        screen_pos: Tuple[float, float],
        now: float,
    ) -> Tuple[GestureState, float, str, Tuple[float, float], Optional[str], Optional[str]]:
        """Handles active modal radial shortcut wheel."""
        if landmarks is None or len(landmarks) < 21:
            return GestureState.RADIAL_MENU, 0.0, "Radial Wheel Active", screen_pos, None, None

        sx, sy = screen_pos
        cx, cy = self.screen_center
        dx = sx - cx
        dy = sy - cy
        dist = math.hypot(dx, dy)
        palm_scale = self._get_palm_scale(landmarks)

        R_INNER = 45.0
        R_OUTER = 160.0

        active_sector = None
        if R_INNER <= dist <= R_OUTER and RADIAL_ACTIONS:
            angle_deg = (math.degrees(math.atan2(cy - sy, dx)) + 360.0) % 360.0
            n_sectors = len(RADIAL_ACTIONS)
            sector_w = 360.0 / n_sectors
            sector_idx = int(((angle_deg + sector_w / 2.0) % 360.0) // sector_w) % n_sectors
            active_sector = RADIAL_ACTIONS[sector_idx]["id"]

        # Check Right Hand Pinch Tap for selection (< 250ms tap)
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        d_pinch = self._euclidean_dist(index_tip, thumb_tip) / palm_scale
        pinch_contact = d_pinch < self.pinch_start_threshold

        if not self.is_pinching:
            if pinch_contact:
                self.is_pinching = True
                self.pinch_start_time = now
        else:
            if d_pinch > self.pinch_release_threshold:
                self.is_pinching = False
                pinch_dur = now - self.pinch_start_time
                if pinch_dur < self.tap_max_time:
                    self.is_radial_active = False
                    self.dual_open_start_time = None
                    if active_sector:
                        return GestureState.RADIAL_MENU, 0.0, f"Executed {active_sector}", screen_pos, active_sector, active_sector
                    else:
                        return GestureState.POINTING, 0.0, "Radial Wheel Dismissed", screen_pos, None, None

        pinch_progress = 1.0 if self.is_pinching else 0.0
        status = f"Radial: {active_sector or 'Neutral'}"
        return GestureState.RADIAL_MENU, pinch_progress, status, screen_pos, None, active_sector

    def _eval_dual_open_activation(
        self,
        landmarks: Optional[List[Any]],
        left_landmarks: Optional[List[Any]],
        screen_pos: Tuple[float, float],
        now: float,
    ) -> Optional[Tuple[GestureState, float, str, Tuple[float, float], Optional[str], Optional[str]]]:
        """Detects dual open palms hold (250ms) to summon the radial menu."""
        if landmarks is None or left_landmarks is None:
            self.dual_open_start_time = None
            return None

        if self._is_hand_fully_open(landmarks) and self._is_hand_fully_open(left_landmarks):
            if self.dual_open_start_time is None:
                self.dual_open_start_time = now
            elif (now - self.dual_open_start_time) >= self.DUAL_OPEN_HOLD_THRESHOLD:
                self.is_radial_active = True
                self.dual_open_start_time = None
                return GestureState.RADIAL_MENU, 0.0, "Radial Menu Opened", screen_pos, None, None
            return GestureState.POINTING, 0.0, "Charging Radial Menu...", screen_pos, None, None
        else:
            self.dual_open_start_time = None
            return None

    def _eval_swipes(
        self,
        landmarks: List[Any],
        screen_pos: Tuple[float, float],
        now: float,
    ) -> Optional[Tuple[GestureState, str, Optional[str]]]:
        """Detects 3-finger swipes for Spaces and Mission Control with debounced hysteresis."""
        # STRICT RULE 1: If hand is in pinch contact (or pinching to click), NEVER evaluate as swipe!
        palm_scale = self._get_palm_scale(landmarks)
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        d_thumb_index = self._euclidean_dist(thumb_tip, index_tip) / palm_scale

        if d_thumb_index < 0.45 or self.is_pinching or self.is_hand_pinched(landmarks):
            self.is_in_swipe_mode = False
            self.three_finger_detect_frames = 0
            self.three_finger_lost_frames = 0
            self.three_finger_history.clear()
            return None

        # STRICT RULE 2: Three fingers (Index, Middle, Ring) MUST ALL be genuinely extended.
        # When pointing with index alone, middle and ring fingers are curled into palm.
        idx_ext = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        mid_ext = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_ext = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP)

        mid_curled = self._is_finger_curled(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_curled = self._is_finger_curled(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP)

        wrist = landmarks[self.WRIST]
        d_wrist_idx = self._euclidean_dist(landmarks[self.INDEX_TIP], wrist)
        d_wrist_mid = self._euclidean_dist(landmarks[self.MIDDLE_TIP], wrist)
        d_wrist_ring = self._euclidean_dist(landmarks[self.RING_TIP], wrist)

        # 3-finger posture: Index, Middle, AND Ring extended outward together,
        # middle & ring not curled into palm, and thumb clearly open away from index.
        is_three_finger_posture = (
            idx_ext and
            mid_ext and
            ring_ext and
            (not mid_curled) and
            (not ring_curled) and
            (d_wrist_mid >= d_wrist_idx * 0.85) and
            (d_wrist_ring >= d_wrist_idx * 0.75) and
            (d_thumb_index > 0.42)
        )

        # Debounced entry: Require 2 consecutive frames to enter swipe mode (prevents 1-frame pointing noise)
        if is_three_finger_posture:
            self.three_finger_detect_frames += 1
            self.three_finger_lost_frames = 0
            if self.three_finger_detect_frames >= 2:
                self.is_in_swipe_mode = True
        else:
            self.three_finger_lost_frames += 1
            if self.three_finger_lost_frames >= 3:
                self.is_in_swipe_mode = False
                self.three_finger_detect_frames = 0
                self.three_finger_history.clear()

        # If confirmed in 3-finger swipe mode:
        if self.is_in_swipe_mode:
            self.cancel_momentum_scroll()
            self.is_pinching = False
            self.is_dragging = False

            # Track Middle MCP (landmark 9) for rock-solid stability during hand translation
            mid_mcp = landmarks[self.MIDDLE_MCP]
            curr_x, curr_y = self._get_lm_xy(mid_mcp)

            self.three_finger_history.append((curr_x, curr_y, now))
            # Maintain rolling 220ms history
            self.three_finger_history = [p for p in self.three_finger_history if (now - p[2]) <= 0.22]

            recent = self.three_finger_history
            if len(recent) < 3:
                return GestureState.SWIPE_NAV, "3-Finger Swipe Ready", None

            # Detect candidate flick strokes starting from the motion inflection point / apex
            candidates = []

            # LEFT candidate (x decreasing from recent maximum)
            p_max_x = max(recent, key=lambda p: p[0])
            dx_left = curr_x - p_max_x[0]
            dt_left = max(now - p_max_x[2], 0.001)
            dy_left = curr_y - p_max_x[1]
            if dx_left < -self.SWIPE_DIST_THRESHOLD and dt_left >= 0.035:
                speed_l = abs(dx_left) / dt_left
                if speed_l >= self.MIN_SWIPE_SPEED and abs(dx_left) > abs(dy_left) * 1.20:
                    candidates.append(("LEFT", speed_l, "Desktop Space Left", "SPACE_RIGHT"))

            # RIGHT candidate (x increasing from recent minimum)
            p_min_x = min(recent, key=lambda p: p[0])
            dx_right = curr_x - p_min_x[0]
            dt_right = max(now - p_min_x[2], 0.001)
            dy_right = curr_y - p_min_x[1]
            if dx_right > self.SWIPE_DIST_THRESHOLD and dt_right >= 0.035:
                speed_r = dx_right / dt_right
                if speed_r >= self.MIN_SWIPE_SPEED and abs(dx_right) > abs(dy_right) * 1.20:
                    candidates.append(("RIGHT", speed_r, "Desktop Space Right", "SPACE_LEFT"))

            # UP candidate (y decreasing from recent maximum)
            p_max_y = max(recent, key=lambda p: p[1])
            dy_up = curr_y - p_max_y[1]
            dt_up = max(now - p_max_y[2], 0.001)
            dx_up = curr_x - p_max_y[0]
            if dy_up < -self.SWIPE_DIST_THRESHOLD and dt_up >= 0.035:
                speed_u = abs(dy_up) / dt_up
                if speed_u >= self.MIN_SWIPE_SPEED and abs(dy_up) > abs(dx_up) * 1.20:
                    candidates.append(("UP", speed_u, "Mission Control", "MISSION_CONTROL"))

            # DOWN candidate (y increasing from recent minimum)
            p_min_y = min(recent, key=lambda p: p[1])
            dy_down = curr_y - p_min_y[1]
            dt_down = max(now - p_min_y[2], 0.001)
            dx_down = curr_x - p_min_y[0]
            if dy_down > self.SWIPE_DIST_THRESHOLD and dt_down >= 0.035:
                speed_d = dy_down / dt_down
                if speed_d >= self.MIN_SWIPE_SPEED and abs(dy_down) > abs(dx_down) * 1.20:
                    candidates.append(("DOWN", speed_d, "App Exposé", "APP_EXPOSE"))

            if not candidates:
                return GestureState.SWIPE_NAV, "3-Finger Swipe Ready", None

            # Pick fastest candidate flick
            candidates.sort(key=lambda c: c[1], reverse=True)
            best_dir, best_speed, best_status, best_action = candidates[0]

            dt_last = now - self.last_swipe_time

            # Absolute minimal hardware debounce
            if dt_last < 0.12:
                return GestureState.SWIPE_NAV, "3-Finger Navigation", None

            if self.last_swipe_dir is not None:
                # Suppress opposite recoil bounce-back
                is_opposite = (
                    (self.last_swipe_dir == "LEFT" and best_dir == "RIGHT") or
                    (self.last_swipe_dir == "RIGHT" and best_dir == "LEFT") or
                    (self.last_swipe_dir == "UP" and best_dir == "DOWN") or
                    (self.last_swipe_dir == "DOWN" and best_dir == "UP")
                )
                if is_opposite and dt_last < self.OPPOSITE_DIR_COOLDOWN:
                    return GestureState.SWIPE_NAV, "3-Finger Navigation", None

                # Same-direction quick re-arm for successive swipes
                is_same = (self.last_swipe_dir == best_dir)
                if is_same and dt_last < self.SAME_DIR_COOLDOWN:
                    return GestureState.SWIPE_NAV, "3-Finger Navigation", None

                # Perpendicular direction cooldown
                is_perp = not is_opposite and not is_same
                if is_perp and dt_last < self.PERP_DIR_COOLDOWN:
                    return GestureState.SWIPE_NAV, "3-Finger Navigation", None

            # Swipe successfully fired!
            self.last_swipe_time = now
            self.last_swipe_dir = best_dir
            self.three_finger_history = [(curr_x, curr_y, now)]
            return GestureState.SWIPE_NAV, best_status, best_action

        return None

    def _eval_scrolling(
        self,
        landmarks: List[Any],
        screen_pos: Tuple[float, float],
        is_modifier_active: bool,
        now: float,
    ) -> Optional[Tuple[GestureState, int, str]]:
        """Handles 2-finger scroll with inertial momentum. Strictly requires Left Hand (modifier) to be up."""
        # STRICT REQUIREMENT: Scrolling ONLY occurs when the Left Hand (modifier) is actively up!
        if not is_modifier_active:
            self.cancel_momentum_scroll()
            return None

        # If hand is in active pinch contact (Right Click), do NOT scroll!
        if self.is_hand_pinched(landmarks) or self.is_pinching:
            self.prev_scroll_y = None
            self.last_scroll_time = None
            self.cancel_momentum_scroll()
            return None

        # Scrolling posture: Index is extended and Middle is extended (or not curled into palm)
        idx_ext = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        mid_ext = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        mid_curled = self._is_finger_curled(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)

        # In 2-finger scroll, Middle finger is extended/up alongside Index. If Middle is curled, user is pointing!
        is_scroll_pose = (idx_ext or mid_ext) and (not mid_curled)

        if is_scroll_pose:
            self.is_momentum_scrolling = False
            self.last_momentum_time = None
            self.is_pinching = False
            self.is_dragging = False

            # Track average Y position of Index and Middle fingertips for rock-solid, jitter-free scrolling
            idx_y = self._get_lm_xy(landmarks[self.INDEX_TIP])[1]
            mid_y = self._get_lm_xy(landmarks[self.MIDDLE_TIP])[1]
            curr_y = (idx_y + mid_y) * 0.5

            if self.prev_scroll_y is None:
                self.prev_scroll_y = curr_y
                self.last_scroll_time = now
                self.scroll_accumulator = 0.0
                return GestureState.SCROLLING, 0, "[SHIFT] Scrolling Engaged"

            dy_raw = curr_y - self.prev_scroll_y
            dt = max(now - (self.last_scroll_time or now), 0.001)

            # Deadband threshold
            if abs(dy_raw) < 0.0015:
                dy_raw = 0.0

            # Accumulate smooth continuous scroll
            self.scroll_accumulator += (-dy_raw * 340.0 * self.scroll_sensitivity)
            self.scroll_velocity_y = (-dy_raw / dt) * 10.0
            self.prev_scroll_y = curr_y
            self.last_scroll_time = now

            scroll_dy_lines = int(self.scroll_accumulator)
            self.scroll_accumulator -= scroll_dy_lines

            status = f"[SHIFT] Scrolling ({'+' if scroll_dy_lines > 0 else ''}{scroll_dy_lines})"
            return GestureState.SCROLLING, scroll_dy_lines, status

        # Hand exited scroll posture (e.g. curled middle finger to point) -> apply momentum decay
        self.prev_scroll_y = None
        self.last_scroll_time = None

        if abs(self.scroll_velocity_y) > self.MOMENTUM_MIN_VELOCITY:
            self.is_momentum_scrolling = True
            if self.last_momentum_time is None:
                self.last_momentum_time = now

            dt_mom = now - self.last_momentum_time
            self.last_momentum_time = now

            decay = math.exp(-self.MOMENTUM_DECAY_RATE * dt_mom)
            self.scroll_velocity_y *= decay

            self.scroll_accumulator += self.scroll_velocity_y * dt_mom * 12.0
            momentum_dy = int(self.scroll_accumulator)
            self.scroll_accumulator -= momentum_dy

            if abs(self.scroll_velocity_y) <= self.MOMENTUM_MIN_VELOCITY or momentum_dy == 0:
                self.cancel_momentum_scroll()
                return None

            return GestureState.SCROLLING, momentum_dy, "[SHIFT] Coasting Scroll..."
        else:
            self.cancel_momentum_scroll()
            return None

    # -------------------------------------------------------------------------
    # Main Process Entry Point
    # -------------------------------------------------------------------------

    def process(
        self,
        landmarks: Optional[List[Any]],
        screen_pos: Tuple[float, float],
        left_landmarks: Optional[List[Any]] = None,
        now: Optional[float] = None,
    ) -> GestureData:
        """
        Processes dominant hand landmarks, screen position, and modifier presence.
        Returns a strongly-typed GestureData object.
        """
        if now is None:
            now = time.time()

        is_modifier_active = (left_landmarks is not None and len(left_landmarks) >= 21)

        # 1. Active Radial Menu
        if self.is_radial_active:
            state, pinch_p, status, eff_pos, nav_act, sec = self._eval_active_radial_menu(
                landmarks, screen_pos, now
            )
            return GestureData(
                screen_x=eff_pos[0],
                screen_y=eff_pos[1],
                state=state,
                pinch_progress=pinch_p,
                hand_landmarks=landmarks,
                is_tracking=True,
                status_message=status,
                is_modifier_active=is_modifier_active,
                nav_action=nav_act,
                radial_sector=sec,
            )

        # 2. Dual Open Palms (Summon Radial Menu)
        radial_summon = self._eval_dual_open_activation(landmarks, left_landmarks, screen_pos, now)
        if radial_summon is not None:
            state, pinch_p, status, eff_pos, nav_act, sec = radial_summon
            return GestureData(
                screen_x=eff_pos[0],
                screen_y=eff_pos[1],
                state=state,
                pinch_progress=pinch_p,
                hand_landmarks=landmarks,
                is_tracking=True,
                status_message=status,
                is_modifier_active=is_modifier_active,
                nav_action=nav_act,
                radial_sector=sec,
            )

        # If right hand is not visible, return idle tracking
        if landmarks is None or len(landmarks) < 21:
            self.cancel_momentum_scroll()
            self.is_pinching = False
            self.is_dragging = False
            status = "[SHIFT] Ready (Show Right Hand)" if is_modifier_active else "Searching for Right Hand..."
            return GestureData(
                screen_x=screen_pos[0],
                screen_y=screen_pos[1],
                state=GestureState.NONE,
                is_tracking=False,
                status_message=status,
                is_modifier_active=is_modifier_active,
            )

        sx, sy = screen_pos

        # 3. Three-Finger Swipes (Spaces & Mission Control)
        # STRICT RULE: 3-finger swipe CANNOT happen when Left Hand is up, or when hand is pinching!
        # Swipes are strictly single-hand navigation gestures with fingers spread open.
        if not is_modifier_active and not self.is_pinching:
            swipe_res = self._eval_swipes(landmarks, screen_pos, now)
            if swipe_res is not None:
                state, status, nav_action = swipe_res
                return GestureData(
                    screen_x=sx,
                    screen_y=sy,
                    state=state,
                    hand_landmarks=landmarks,
                    is_tracking=True,
                    status_message=status,
                    is_modifier_active=is_modifier_active,
                    nav_action=nav_action,
                )
        else:
            self.is_in_swipe_mode = False
            self.three_finger_history.clear()
            self.three_finger_detect_frames = 0
            self.three_finger_lost_frames = 0
            self.last_swipe_dir = None

        # 4. Two-Finger Scrolling & Momentum
        scroll_res = self._eval_scrolling(landmarks, screen_pos, is_modifier_active, now)
        if scroll_res is not None:
            state, scroll_dy, status = scroll_res
            return GestureData(
                screen_x=sx,
                screen_y=sy,
                state=state,
                scroll_dy=scroll_dy,
                hand_landmarks=landmarks,
                is_tracking=True,
                status_message=status,
                is_modifier_active=is_modifier_active,
            )

        # 5. Pointer Navigation & Pinch Clicks / Instant Dragging
        palm_scale = self._get_palm_scale(landmarks)
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        index_dip = landmarks[self.INDEX_DIP]

        d_tip = self._euclidean_dist(index_tip, thumb_tip) / palm_scale
        d_dip = self._euclidean_dist(index_dip, thumb_tip) / palm_scale
        index_thumb_min_dist = min(d_tip, d_dip)

        pinch_contact = index_thumb_min_dist < self.pinch_start_threshold
        raw_progress = 1.0 - (index_thumb_min_dist - self.pinch_start_threshold) / (self.pinch_release_threshold - self.pinch_start_threshold)
        pinch_progress = max(0.0, min(1.0, raw_progress))

        current_action = GestureState.POINTING
        status_msg = "Pointing"
        effective_x, effective_y = sx, sy

        if not self.is_pinching:
            if pinch_contact:
                self.is_pinching = True
                self.is_dragging = False
                self.pinch_start_time = now
                self.pinch_freeze_x = sx
                self.pinch_freeze_y = sy
                self.pinch_start_modifier = is_modifier_active
                current_action = GestureState.PINCHING
                status_msg = "[SHIFT] Pinch Active" if is_modifier_active else "Pinch Active"
                effective_x, effective_y = self.pinch_freeze_x, self.pinch_freeze_y
        else:
            dist_moved = math.hypot(sx - self.pinch_freeze_x, sy - self.pinch_freeze_y)
            pinch_dur = now - self.pinch_start_time

            if index_thumb_min_dist > self.pinch_release_threshold:
                # --- PINCH RELEASED ---
                was_dragging = self.is_dragging
                self.is_pinching = False
                self.is_dragging = False

                # If drag was engaged, always release drag cleanly (even if hand returned near origin)
                if was_dragging:
                    # Drag completed and released
                    current_action = GestureState.POINTING
                    status_msg = "Drag Released"
                    effective_x, effective_y = sx, sy
                else:
                    # Released without initiating a drag -> Trigger Click
                    if self.pinch_start_modifier or is_modifier_active:
                        current_action = GestureState.RIGHT_CLICK
                        status_msg = "[SHIFT] Right Click"
                    else:
                        time_since_last_tap = now - self.last_tap_time
                        tap_dist = math.hypot(self.pinch_freeze_x - self.last_tap_x, self.pinch_freeze_y - self.last_tap_y)
                        if time_since_last_tap < self.double_click_window and tap_dist < self.double_click_dist:
                            current_action = GestureState.DOUBLE_CLICK
                            status_msg = "Double Click"
                            self.last_tap_time = 0.0
                        else:
                            current_action = GestureState.CLICK
                            status_msg = "Left Click"
                            self.last_tap_time = now
                            self.last_tap_x = self.pinch_freeze_x
                            self.last_tap_y = self.pinch_freeze_y

                    effective_x, effective_y = self.pinch_freeze_x, self.pinch_freeze_y
            else:
                # --- PINCH HELD ---
                # STRICT RULE: Dragging is ONLY allowed when Left Hand modifier is NOT up!
                if not is_modifier_active and not self.pinch_start_modifier:
                    # Deliberate spatial drag: requires travel > 38px, OR sustained hold > 350ms AND movement > 28px
                    if dist_moved > self.drag_dist_threshold or (pinch_dur > 0.35 and dist_moved > 28.0):
                        self.is_dragging = True

                if self.is_dragging:
                    current_action = GestureState.DRAGGING
                    status_msg = "Dragging"
                    effective_x, effective_y = sx, sy
                else:
                    current_action = GestureState.PINCHING
                    status_msg = "[SHIFT] Pinch Active" if is_modifier_active else "Pinch Active"
                    effective_x, effective_y = self.pinch_freeze_x, self.pinch_freeze_y

        return GestureData(
            raw_x=sx,
            raw_y=sy,
            screen_x=effective_x,
            screen_y=effective_y,
            state=current_action,
            pinch_progress=pinch_progress,
            hand_landmarks=landmarks,
            is_tracking=True,
            status_message=status_msg,
            is_modifier_active=is_modifier_active,
            drag_start_x=self.pinch_freeze_x,
            drag_start_y=self.pinch_freeze_y,
        )
