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
        self.pinch_start_threshold = 0.40
        self.pinch_release_threshold = 0.54

        # Timing and travel thresholds
        self.tap_max_time = 0.35          # Tap duration < 350ms
        self.drag_dist_threshold = 12.0   # Crisp 12px spatial travel threshold (Zero time delay)
        self.double_click_window = 0.40   # Double-tap window (400ms)

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

        # Three-finger space navigation
        self.three_finger_history: List[Tuple[float, float, float]] = []
        self.last_swipe_time: float = 0.0
        self.swipe_cooldown: float = 0.45

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
        if landmarks is None or len(landmarks) < 21:
            return False
        wrist = landmarks[self.WRIST]
        d_tip_wrist = self._euclidean_dist(landmarks[tip_idx], wrist)
        d_pip_wrist = self._euclidean_dist(landmarks[pip_idx], wrist)
        if mcp_idx is not None:
            d_tip_mcp = self._euclidean_dist(landmarks[tip_idx], landmarks[mcp_idx])
            d_pip_mcp = self._euclidean_dist(landmarks[pip_idx], landmarks[mcp_idx])
            return (d_tip_wrist > d_pip_wrist * 1.05) and (d_tip_mcp > d_pip_mcp * 1.15)
        return d_tip_wrist > d_pip_wrist * 1.05

    def _is_finger_curled(self, landmarks, tip_idx: int, pip_idx: int, mcp_idx: Optional[int] = None) -> bool:
        if landmarks is None or len(landmarks) < 21:
            return False
        wrist = landmarks[self.WRIST]
        d_tip_wrist = self._euclidean_dist(landmarks[tip_idx], wrist)
        d_pip_wrist = self._euclidean_dist(landmarks[pip_idx], wrist)
        if mcp_idx is not None:
            d_tip_mcp = self._euclidean_dist(landmarks[tip_idx], landmarks[mcp_idx])
            d_pip_mcp = self._euclidean_dist(landmarks[pip_idx], landmarks[mcp_idx])
            return (d_tip_wrist < d_pip_wrist) or (d_tip_mcp < d_pip_mcp * 1.05)
        return d_tip_wrist < d_pip_wrist

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

    def is_hand_pinched(self, landmarks) -> bool:
        """Evaluates if hand configuration is in an active pinch contact."""
        if landmarks is None or len(landmarks) < 21:
            return False
        palm_scale = self._get_palm_scale(landmarks)
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        index_dip = landmarks[self.INDEX_DIP]

        d_tip = self._euclidean_dist(index_tip, thumb_tip) / palm_scale
        d_dip = self._euclidean_dist(index_dip, thumb_tip) / palm_scale
        return min(d_tip, d_dip) < self.pinch_start_threshold

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
        """Detects 3-finger swipes for Spaces and Mission Control."""
        idx_ext = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        mid_ext = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_ext = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP)
        pinky_ext = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)

        # 3 fingers extended (Index, Middle, Ring), Pinky NOT extended
        if idx_ext and mid_ext and ring_ext and not pinky_ext:
            # Active in 3-finger posture: cancel any scrolling momentum immediately
            self.cancel_momentum_scroll()

            # Track Middle MCP (landmark 9) for rock-solid stability during hand translation
            mid_mcp = landmarks[self.MIDDLE_MCP]
            curr_x, curr_y = self._get_lm_xy(mid_mcp)
            self.three_finger_history.append((curr_x, curr_y, now))

            # Maintain rolling 0.35s history
            self.three_finger_history = [p for p in self.three_finger_history if (now - p[2]) <= 0.35]

            # Cooldown check
            if (now - self.last_swipe_time) > self.swipe_cooldown and len(self.three_finger_history) >= 3:
                start_x, start_y, _ = self.three_finger_history[0]
                dx = curr_x - start_x
                dy = curr_y - start_y

                # Responsive threshold (7.5% of frame distance)
                SWIPE_DIST_THRESHOLD = 0.075

                if abs(dx) > abs(dy) and abs(dx) > SWIPE_DIST_THRESHOLD:
                    self.last_swipe_time = now
                    self.three_finger_history.clear()
                    # Inverted per user experience:
                    # Swiping left on screen (dx < 0) -> Space Right
                    # Swiping right on screen (dx > 0) -> Space Left
                    if dx < 0:
                        return GestureState.SWIPE_NAV, "Desktop Space Left", "SPACE_RIGHT"
                    else:
                        return GestureState.SWIPE_NAV, "Desktop Space Right", "SPACE_LEFT"

                elif abs(dy) > abs(dx) and abs(dy) > SWIPE_DIST_THRESHOLD:
                    self.last_swipe_time = now
                    self.three_finger_history.clear()
                    if dy < 0:
                        # Hand swiped UP (y decreased) -> Mission Control
                        return GestureState.SWIPE_NAV, "Mission Control", "MISSION_CONTROL"
                    else:
                        # Hand swiped DOWN (y increased) -> App Exposé
                        return GestureState.SWIPE_NAV, "App Exposé", "APP_EXPOSE"

            # 3 fingers are held up: return SWIPE_NAV so it does NOT leak into scrolling or clicking!
            return GestureState.SWIPE_NAV, "3-Finger Swipe Ready", None
        else:
            self.three_finger_history.clear()
            return None

    def _eval_scrolling(
        self,
        landmarks: List[Any],
        screen_pos: Tuple[float, float],
        is_modifier_active: bool,
        now: float,
    ) -> Optional[Tuple[GestureState, int, str]]:
        """Handles 2-finger scroll with inertial momentum."""
        idx_ext = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        mid_ext = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_ext = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP)

        # STRICT EXCLUSION: If Ring finger is extended, this is 3-finger swipe mode, NOT 2-finger scroll!
        if ring_ext:
            self.cancel_momentum_scroll()
            return None

        # Two-finger scroll: Index + Middle extended, Ring NOT extended
        if idx_ext and mid_ext and not ring_ext:
            self.is_momentum_scrolling = False
            self.last_momentum_time = None

            mid_tip = landmarks[self.MIDDLE_TIP]
            _, mid_y = self._get_lm_xy(mid_tip)

            if self.prev_scroll_y is None:
                self.prev_scroll_y = mid_y
                self.last_scroll_time = now
                self.scroll_accumulator = 0.0
                return GestureState.SCROLLING, 0, "Scrolling Engaged"

            dy_raw = mid_y - self.prev_scroll_y
            dt = max(now - (self.last_scroll_time or now), 0.001)

            # Deadband threshold
            if abs(dy_raw) < 0.003:
                self.prev_scroll_y = mid_y
                self.last_scroll_time = now
                return GestureState.SCROLLING, 0, "Scrolling Neutral"

            # Compute velocity and accumulate
            scroll_dy_lines = int(-dy_raw * 180.0 * self.scroll_sensitivity)
            self.scroll_velocity_y = (-dy_raw / dt) * 12.0
            self.prev_scroll_y = mid_y
            self.last_scroll_time = now

            status = f"Scrolling ({'+' if scroll_dy_lines > 0 else ''}{scroll_dy_lines})"
            return GestureState.SCROLLING, scroll_dy_lines, status

        # Hand exited scroll posture -> apply momentum decay
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

            momentum_dy = int(self.scroll_velocity_y * dt_mom * 8.0)
            if abs(self.scroll_velocity_y) <= self.MOMENTUM_MIN_VELOCITY or momentum_dy == 0:
                self.cancel_momentum_scroll()
                return None

            return GestureState.SCROLLING, momentum_dy, "Coasting Scroll..."
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
        # Priority check: evaluated BEFORE scrolling and pointer actions
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

            if index_thumb_min_dist > self.pinch_release_threshold:
                # --- PINCH RELEASED ---
                was_dragging = self.is_dragging
                self.is_pinching = False
                self.is_dragging = False

                if was_dragging:
                    # Drag completed and released
                    current_action = GestureState.POINTING
                    status_msg = "Drag Released"
                    effective_x, effective_y = sx, sy
                else:
                    # Released without travel -> Trigger Click
                    if self.pinch_start_modifier or is_modifier_active:
                        current_action = GestureState.RIGHT_CLICK
                        status_msg = "[SHIFT] Right Click"
                    else:
                        time_since_last_tap = now - self.last_tap_time
                        tap_dist = math.hypot(self.pinch_freeze_x - self.last_tap_x, self.pinch_freeze_y - self.last_tap_y)
                        if time_since_last_tap < self.double_click_window and tap_dist < 45.0:
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
                # Instant spatial threshold: ZERO time delay!
                if dist_moved > self.drag_dist_threshold:
                    self.is_dragging = True

                if self.is_dragging:
                    current_action = GestureState.DRAGGING
                    status_msg = "Dragging"
                    effective_x, effective_y = sx, sy
                else:
                    current_action = GestureState.PINCHING
                    status_msg = "Pinch Active"
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
