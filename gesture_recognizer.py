"""
Dual-Hand Modifier Gesture Recognition Engine for FRIDAY with Strict Precedence Gating.
- Dominant Hand (Right Hand): Drives pointer navigation, pinch clicks, and two-finger scrolling.
- Modifier Hand (Left Hand): Acts as a physical "Shift Key" when raised in frame.
- Strict Precedence Gating:
    1. Extension Mode Checked FIRST: If Index + Middle are extended -> Scroll Mode ONLY (bypasses all clicks).
    2. Right-Click Condition: ONLY evaluated if Middle Finger is CURLED and Left Hand is raised.
"""

import math
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


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
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    RADIAL_MENU = "RADIAL MENU"
    AI_LISTENING = "AI LISTENING"
    SNIP_DRAG = "SNIP DRAG"
    SNIP_RELEASE = "SNIP RELEASE"


class GestureData:
    """Carries full state information of the detected gesture per frame."""

    def __init__(
        self,
        raw_x: float = 0.0,
        raw_y: float = 0.0,
        screen_x: float = 0.0,
        screen_y: float = 0.0,
        state: GestureState = GestureState.NONE,
        pinch_progress: float = 0.0,
        scroll_dy: int = 0,
        hand_landmarks: Optional[List[Any]] = None,
        is_tracking: bool = False,
        confidence: float = 0.0,
        fps: float = 0.0,
        status_message: str = "Ready",
        is_modifier_active: bool = False,
        nav_action: Optional[str] = None,
        radial_sector: Optional[str] = None,
        snip_box: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.state = state
        self.pinch_progress = pinch_progress
        self.scroll_dy = scroll_dy
        self.hand_landmarks = hand_landmarks
        self.is_tracking = is_tracking
        self.confidence = confidence
        self.fps = fps
        self.status_message = status_message
        self.is_modifier_active = is_modifier_active
        self.nav_action = nav_action
        self.radial_sector = radial_sector
        self.snip_box = snip_box


class GestureRecognizer:
    """Analyzes 3D MediaPipe landmarks with robust two-finger scroll, dual-hand modifier, and radial shortcut wheel."""

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
        # Scale-Invariant & Contact Thresholds (Hardcoded Optimal Tracking Constants)
        self.pinch_start_threshold = 0.40      # Optimal pinch contact threshold
        self.pinch_release_threshold = 0.54

        # Timing and spatial movement thresholds
        self.tap_max_time = 0.25               # Quick tap < 250ms
        self.drag_time_threshold = 0.45        # 450ms hold delay before drag engages
        self.drag_dist_threshold = 35.0        # Strict 35px travel required before drag engages
        self.double_click_window = 0.40        # 400ms double-tap window

        # State tracking
        self.is_pinching = False
        self.is_dragging = False

        self.pinch_start_time: float = 0.0
        self.pinch_freeze_x: float = 0.0
        self.pinch_freeze_y: float = 0.0
        self.pinch_start_modifier: bool = False
        self.is_snipping: bool = False
        self.snip_box: Optional[Tuple[float, float, float, float]] = None

        self.last_tap_time: float = 0.0
        self.last_tap_x: float = 0.0
        self.last_tap_y: float = 0.0

        # Kinematic velocity tracking
        self.prev_knuckle_pos: Optional[Tuple[float, float]] = None
        self.prev_tip_pos: Optional[Tuple[float, float]] = None
        self.prev_time: Optional[float] = None

        # Two-finger scroll tracking & Inertial Momentum
        self.prev_scroll_y: Optional[float] = None
        self.scroll_accumulator: float = 0.0
        self.scroll_sensitivity = 1.0
        self.scroll_velocity_y: float = 0.0
        self.is_momentum_scrolling: bool = False
        self.last_scroll_time: Optional[float] = None
        self.last_momentum_time: Optional[float] = None
        self.MOMENTUM_MIN_VELOCITY: float = 3.0
        self.MOMENTUM_DECAY_RATE: float = 6.5

        # Three-finger space navigation & Mission Control tracking
        self.three_finger_history: List[Tuple[float, float, float]] = []
        self.last_swipe_time: float = 0.0
        self.swipe_cooldown: float = 0.60

        # Push-to-talk voice dictation fist hold debounce (Single Left Fist)
        self.fist_start_time: Optional[float] = None
        self.is_fist_active: bool = False
        self.FIST_HOLD_THRESHOLD: float = 0.5  # 500ms continuous hold required

        # Dual-Fist AI Assistant ("Ask FRIDAY") hold debounce
        self.dual_fist_start_time: Optional[float] = None
        self.is_dual_fist_active: bool = False
        self.DUAL_FIST_HOLD_THRESHOLD: float = 0.4  # 400ms continuous hold required

        # GTA-Style Radial Shortcut Wheel (Dual Open Palms)
        self.dual_open_start_time: Optional[float] = None
        self.is_radial_active: bool = False
        self.DUAL_OPEN_HOLD_THRESHOLD: float = 0.25  # 250ms continuous hold required
        self.screen_center: Tuple[float, float] = (1470.0 / 2.0, 956.0 / 2.0)

    def set_screen_dimensions(self, width: float, height: float):
        """Sets the screen center coordinates for radial wheel calculations."""
        self.screen_center = (width / 2.0, height / 2.0)

    def dismiss_radial_menu(self):
        """Manually dismisses the radial menu modal state."""
        self.is_radial_active = False
        self.dual_open_start_time = None

    def _euclidean_dist(self, p1, p2) -> float:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = getattr(p1, 'z', 0.0) - getattr(p2, 'z', 0.0)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _get_palm_scale(self, landmarks) -> float:
        """Computes reference scale based on wrist to middle MCP and palm breadth."""
        wrist = landmarks[self.WRIST]
        middle_mcp = landmarks[self.MIDDLE_MCP]
        index_mcp = landmarks[self.INDEX_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]

        wrist_to_middle = self._euclidean_dist(wrist, middle_mcp)
        palm_breadth = self._euclidean_dist(index_mcp, pinky_mcp)
        scale = max((wrist_to_middle + palm_breadth) * 0.5, 0.01)
        return scale

    def _is_finger_extended(self, landmarks, tip_idx: int, pip_idx: int, mcp_idx: Optional[int] = None) -> bool:
        """Returns True if the finger is extended (tilt and rotation invariant)."""
        if landmarks is None or len(landmarks) < 21:
            return False
        wrist = landmarks[self.WRIST]
        d_tip_wrist = self._euclidean_dist(landmarks[tip_idx], wrist)
        d_pip_wrist = self._euclidean_dist(landmarks[pip_idx], wrist)
        if mcp_idx is not None:
            d_mcp_wrist = self._euclidean_dist(landmarks[mcp_idx], wrist)
            return (d_tip_wrist > d_pip_wrist * 1.06) and (d_tip_wrist > d_mcp_wrist * 1.12)
        return d_tip_wrist > d_pip_wrist * 1.06

    def _is_finger_curled(self, landmarks, tip_idx: int, pip_idx: int, mcp_idx: Optional[int] = None) -> bool:
        """Returns True if the finger tip is curled close to palm/wrist (tilt and rotation invariant)."""
        if landmarks is None or len(landmarks) < 21:
            return False
        wrist = landmarks[self.WRIST]
        d_tip_wrist = self._euclidean_dist(landmarks[tip_idx], wrist)
        d_pip_wrist = self._euclidean_dist(landmarks[pip_idx], wrist)
        if mcp_idx is not None:
            d_tip_mcp = self._euclidean_dist(landmarks[tip_idx], landmarks[mcp_idx])
            d_pip_mcp = self._euclidean_dist(landmarks[pip_idx], landmarks[mcp_idx])
            return (d_tip_wrist <= d_pip_wrist * 1.15) or (d_tip_mcp <= d_pip_mcp * 1.20)
        return d_tip_wrist <= d_pip_wrist * 1.15

    def _is_thumb_extended(self, landmarks) -> bool:
        """Returns True if the thumb is extended away from wrist and palm (tilt and rotation invariant)."""
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
        """Returns True if all 5 fingers of the hand are extended (tilt and rotation invariant)."""
        if landmarks is None or len(landmarks) < 21:
            return False
        thumb_ext = self._is_thumb_extended(landmarks)
        idx_ext = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        mid_ext = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        ring_ext = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP)
        pinky_ext = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)
        return thumb_ext and idx_ext and mid_ext and ring_ext and pinky_ext

    def _is_hand_fist(self, landmarks) -> bool:
        """Returns True if all 4 fingers (Index, Middle, Ring, Pinky) are curled into a fist (tilt-invariant)."""
        if landmarks is None or len(landmarks) < 21:
            return False
        for tip_idx, pip_idx, mcp_idx in [
            (self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            (self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            (self.RING_TIP, self.RING_PIP, self.RING_MCP),
            (self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP),
        ]:
            if not self._is_finger_curled(landmarks, tip_idx, pip_idx, mcp_idx):
                return False
        return True

    def _is_left_hand_fist(self, left_landmarks) -> bool:
        """Returns True if all 4 fingers of the left hand are curled into a fist for Push-to-Talk."""
        return self._is_hand_fist(left_landmarks)

    def _is_left_hand_extended(self, left_landmarks) -> bool:
        """Returns True if left hand fingers are extended for Modifier (Shift) mode (tilt-invariant)."""
        if left_landmarks is None or len(left_landmarks) < 21:
            return False
        idx_ext = self._is_finger_extended(left_landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        mid_ext = self._is_finger_extended(left_landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        return idx_ext and mid_ext

    def cancel_momentum_scroll(self):
        """Immediately cancels ongoing inertial momentum scrolling."""
        self.is_momentum_scrolling = False
        self.scroll_velocity_y = 0.0
        self.scroll_accumulator = 0.0
        self.last_momentum_time = None

    def get_stable_pointer_coords(self, landmarks) -> Tuple[float, float]:
        """
        Drives cursor navigation directly from Index MCP (Landmark 5, base knuckle).
        Does NOT move when fingers flex or tap during clicks.
        """
        mcp5 = landmarks[self.INDEX_MCP]
        return mcp5.x, mcp5.y

    def process(
        self,
        landmarks: Optional[List[Any]],
        screen_pos: Tuple[float, float],
        is_left_hand_present: bool = False,
        now: Optional[float] = None,
        left_landmarks: Optional[List[Any]] = None,
    ) -> Tuple[GestureState, float, int, str, Tuple[float, float], Optional[str], Optional[str]]:
        """
        Processes dominant hand landmarks, screen position, and modifier presence with strict 5-tier hierarchy.
        Returns:
            (gesture_state, pinch_progress, scroll_dy, status_message, effective_screen_pos, nav_action, radial_sector)
        """
        if now is None:
            now = time.time()

        # =========================================================================
        # PRIORITY 1: DUAL OPEN PALMS (RADIAL MENU SUMMON & STICKY MODAL STATE)
        # =========================================================================
        # 1A. If already active: LOCKED OPEN MODAL (never auto-close when hands move or drop)
        if self.is_radial_active:
            if landmarks is None or len(landmarks) < 21:
                # Hands lowered / out of frame -> radial menu stays visible and locked open!
                return GestureState.RADIAL_MENU, 0.0, 0, "Radial Wheel Active", screen_pos, None, None, None

            sx, sy = screen_pos
            cx, cy = self.screen_center
            dx = sx - cx
            dy = sy - cy
            dist = math.hypot(dx, dy)
            palm_scale = self._get_palm_scale(landmarks)

            # Strict radial bounds (inner deadzone 45px, outer limit 160px)
            R_INNER = 45.0
            R_OUTER = 160.0

            active_sector = None
            if R_INNER <= dist <= R_OUTER:
                deg = math.degrees(math.atan2(dy, dx))
                if -45.0 <= deg <= 45.0:
                    active_sector = "NEW_TAB"
                elif 45.0 < deg <= 135.0:
                    active_sector = "ESCAPE"
                elif -135.0 <= deg < -45.0:
                    active_sector = "ENTER"
                else:
                    active_sector = "CLOSE_TAB"

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
                        # Pinch tap completed -> dismiss modal state
                        self.is_radial_active = False

                        # 1. Inside valid sector: Execute corresponding shortcut
                        if R_INNER <= dist <= R_OUTER and active_sector is not None:
                            nav_action = active_sector
                            return GestureState.RADIAL_MENU, 0.0, 0, f"Executed {nav_action}", screen_pos, nav_action, active_sector, None
                        else:
                            # 2. Outside wheel or in deadzone: Dismiss without action
                            return GestureState.POINTING, 0.0, 0, "Radial Wheel Dismissed", screen_pos, None, None, None

            return GestureState.RADIAL_MENU, (1.0 if self.is_pinching else 0.0), 0, f"Radial: {active_sector or 'Neutral'}", screen_pos, None, active_sector, None

        # 1B. Dual Open Palms Detection (Charge for 250ms & suppress all other gestures)
        if left_landmarks is not None and landmarks is not None:
            is_left_open = self._is_hand_fully_open(left_landmarks)
            is_right_open = self._is_hand_fully_open(landmarks)
            if is_left_open and is_right_open:
                if self.dual_open_start_time is None:
                    self.dual_open_start_time = now
                if (now - self.dual_open_start_time) >= self.DUAL_OPEN_HOLD_THRESHOLD:
                    self.is_radial_active = True
                    self.reset_pinch_states()
                    return GestureState.RADIAL_MENU, 0.0, 0, "Radial Menu Opened", screen_pos, None, None, None
                else:
                    # While charging dual open palms, SUPPRESS all lower priority gestures (scroll, clicks, drags)
                    self.reset_pinch_states()
                    self.prev_scroll_y = None
                    self.scroll_accumulator = 0.0
                    return GestureState.POINTING, 0.0, 0, "Charging Radial Menu...", screen_pos, None, None, None
            else:
                self.dual_open_start_time = None
        else:
            self.dual_open_start_time = None

        # =========================================================================
        # PRIORITY 3: LEFT-HAND PUSH-TO-TALK (VOICE DICTATION & "FRIDAY" ROUTING)
        # =========================================================================
        is_left_fist = False
        is_left_extended = False

        if left_landmarks is not None and len(left_landmarks) >= 21:
            is_left_fist = self._is_left_hand_fist(left_landmarks)
            if not is_left_fist:
                is_left_extended = self._is_left_hand_extended(left_landmarks)

        # Single Left-Hand Fist or Extended Push-to-Talk Trigger
        if is_left_fist:
            if self.fist_start_time is None:
                self.fist_start_time = now

            elapsed_single = now - self.fist_start_time
            if elapsed_single >= self.FIST_HOLD_THRESHOLD:
                self.is_fist_active = True
                self.reset_pinch_states()
                self.prev_scroll_y = None
                self.scroll_accumulator = 0.0
                return GestureState.LISTENING, 0.0, 0, "● Push-to-Talk (Listening...)", screen_pos, None, None, None
            else:
                is_left_extended = False
        else:
            self.fist_start_time = None
            self.is_fist_active = False

        # =========================================================================
        # NO RIGHT HAND CHECK (Zero cursor, clicks, or pinches)
        # =========================================================================
        if landmarks is None or len(landmarks) < 21:
            self.reset_right_hand_states()
            if is_left_fist and self.is_fist_active:
                return GestureState.LISTENING, 0.0, 0, "● Push-to-Talk (Listening...)", screen_pos, None, None, None
            elif is_left_fist:
                return GestureState.NONE, 0.0, 0, "Left Fist (Hold for Voice)...", screen_pos, None, None, None
            elif is_left_extended:
                return GestureState.NONE, 0.0, 0, "[SHIFT] Modifier Ready (Show Right Hand)", screen_pos, None, None, None
            else:
                return GestureState.NONE, 0.0, 0, "Searching for Right Hand...", screen_pos, None, None, None

        # Modifier is active only when Left Hand is extended and NOT a fist
        is_modifier_active = is_left_extended and not is_left_fist

        palm_scale = self._get_palm_scale(landmarks)
        sx, sy = screen_pos

        # Kinematics Update: Calculate knuckle and tip velocities
        knuckle = landmarks[self.INDEX_MCP]
        tip = landmarks[self.INDEX_TIP]
        is_stationary_knuckle = False

        if self.prev_knuckle_pos is not None and self.prev_time is not None:
            dt = max(now - self.prev_time, 0.001)
            v_knuckle = math.hypot(knuckle.x - self.prev_knuckle_pos[0], knuckle.y - self.prev_knuckle_pos[1]) / dt
            v_tip = math.hypot(tip.x - self.prev_tip_pos[0], tip.y - self.prev_tip_pos[1]) / dt
            if v_knuckle < 0.18 and v_tip > 0.05:
                is_stationary_knuckle = True

        self.prev_knuckle_pos = (knuckle.x, knuckle.y)
        self.prev_tip_pos = (tip.x, tip.y)
        self.prev_time = now

        wrist = landmarks[self.WRIST]
        is_index_extended = self._is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP)
        is_middle_extended = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)
        is_ring_extended = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP)
        is_pinky_extended = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)
        is_ring_curled = self._is_finger_curled(landmarks, self.RING_TIP, self.RING_PIP, self.RING_MCP)
        is_pinky_curled = self._is_finger_curled(landmarks, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP)

        # =========================================================================
        # PRIORITY 2: 3-FINGER SPATIAL NAVIGATION (Index + Middle + Ring Extended, Pinky Curled)
        # =========================================================================
        is_three_finger = (not is_left_fist) and is_index_extended and is_middle_extended and is_ring_extended and is_pinky_curled

        if is_three_finger:
            self.reset_pinch_states()
            self.cancel_momentum_scroll()
            self.prev_scroll_y = None
            self.scroll_accumulator = 0.0

            # Centroid of 3 fingertips
            c_tip_x = (landmarks[self.INDEX_TIP].x + landmarks[self.MIDDLE_TIP].x + landmarks[self.RING_TIP].x) / 3.0
            c_tip_y = (landmarks[self.INDEX_TIP].y + landmarks[self.MIDDLE_TIP].y + landmarks[self.RING_TIP].y) / 3.0

            self.three_finger_history.append((c_tip_x, c_tip_y, now))
            self.three_finger_history = [p for p in self.three_finger_history if (now - p[2]) <= 0.40][-8:]

            nav_action = None
            status_msg = "3-Finger Nav Active"

            if len(self.three_finger_history) >= 2 and (now - self.last_swipe_time) >= self.swipe_cooldown:
                start_x, start_y, _ = self.three_finger_history[0]
                dx = c_tip_x - start_x
                dy = c_tip_y - start_y
                abs_dx = abs(dx)
                abs_dy = abs(dy)

                # 1. Horizontal Swipe Check (Space Switch)
                if abs_dx > 0.08 and abs_dx > 1.4 * abs_dy:
                    if dx < 0:
                        nav_action = "SPACE_RIGHT"
                        status_msg = "Swipe Left ➔ Space Right"
                    else:
                        nav_action = "SPACE_LEFT"
                        status_msg = "Swipe Right ➔ Space Left"
                    self.last_swipe_time = now
                    self.three_finger_history.clear()

                # 2. Vertical Swipe Up Check (Mission Control)
                elif dy < -0.07 and abs_dy > 1.3 * abs_dx:
                    nav_action = "MISSION_CONTROL"
                    status_msg = "Swipe Up ➔ Mission Control"
                    self.last_swipe_time = now
                    self.three_finger_history.clear()

            return GestureState.SWIPE_NAV, 0.0, 0, status_msg, screen_pos, nav_action, None, None

        self.three_finger_history.clear()

        # =========================================================================
        # PRIORITY 4: STRICT TWO-FINGER SCROLL (Index + Middle Extended, Ring AND Pinky Curled)
        # =========================================================================
        is_two_finger = is_index_extended and is_middle_extended and is_ring_curled and is_pinky_curled

        if is_two_finger:
            self.reset_pinch_states()

            if is_modifier_active:
                curr_y = (landmarks[self.INDEX_TIP].y + landmarks[self.MIDDLE_TIP].y) * 0.5
                scroll_dy = 0

                if self.prev_scroll_y is not None:
                    dt = max(now - (self.last_scroll_time or now), 0.001)
                    delta_y = (self.prev_scroll_y - curr_y) * 140.0 * self.scroll_sensitivity
                    self.scroll_accumulator += delta_y
                    instant_vel = delta_y / max(dt, 0.016)
                    self.scroll_velocity_y = 0.55 * self.scroll_velocity_y + 0.45 * instant_vel
                    if abs(self.scroll_accumulator) >= 1.0:
                        scroll_dy = int(self.scroll_accumulator)
                        self.scroll_accumulator -= scroll_dy

                self.prev_scroll_y = curr_y
                self.last_scroll_time = now
                self.is_momentum_scrolling = False

                if scroll_dy > 0:
                    status_msg = f"Scrolling Up (+{scroll_dy})"
                elif scroll_dy < 0:
                    status_msg = f"Scrolling Down ({scroll_dy})"
                else:
                    status_msg = "[SHIFT] Scroll Active (Move Up/Down)"

                return GestureState.SCROLLING, 0.0, scroll_dy, status_msg, screen_pos, None, None, None
            else:
                if self.prev_scroll_y is not None and abs(self.scroll_velocity_y) > self.MOMENTUM_MIN_VELOCITY:
                    self.is_momentum_scrolling = True
                    self.last_momentum_time = now
                self.prev_scroll_y = None
                self.scroll_accumulator = 0.0
                return GestureState.POINTING, 0.0, 0, "Two-Finger Neutral (Extend Left Hand to Scroll)", screen_pos, None, None, None

        # Exiting active scroll -> check if momentum should kick in
        if self.prev_scroll_y is not None:
            if abs(self.scroll_velocity_y) > self.MOMENTUM_MIN_VELOCITY:
                self.is_momentum_scrolling = True
                self.last_momentum_time = now
            self.prev_scroll_y = None

        # =========================================================================
        # PRIORITY 5: 1-FINGER POINTING & PINCH CLICK DETECTION (Middle Finger Curled)
        # =========================================================================
        middle_curled = self._is_finger_curled(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP)


        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        index_dip = landmarks[self.INDEX_DIP]

        d_tip = self._euclidean_dist(index_tip, thumb_tip) / palm_scale
        d_dip = self._euclidean_dist(index_dip, thumb_tip) / palm_scale
        index_thumb_min_dist = min(d_tip, d_dip)

        open_dist = self.pinch_release_threshold * 1.35
        close_dist = self.pinch_start_threshold
        pinch_progress = max(0.0, min(1.0, (open_dist - index_thumb_min_dist) / max(open_dist - close_dist, 0.01)))

        current_action = GestureState.POINTING
        status_msg = "Pointing / Hover"
        if is_modifier_active:
            status_msg = "[SHIFT] Modifier Active"
        effective_pos = (sx, sy)

        # Pinch intent requires middle finger to be curled and left hand NOT in a fist
        pinch_intent = (not is_left_fist) and middle_curled and (
            index_thumb_min_dist < self.pinch_start_threshold or
            (is_stationary_knuckle and index_thumb_min_dist < (self.pinch_start_threshold * 1.15))
        )

        if pinch_intent or self.is_pinching:
            self.cancel_momentum_scroll()

        # Handle Inertial Momentum Coasting when pointing
        if not self.is_pinching and self.is_momentum_scrolling:
            dt = max(now - (self.last_momentum_time or now), 0.001)
            self.last_momentum_time = now
            decay = math.exp(-self.MOMENTUM_DECAY_RATE * dt)
            self.scroll_velocity_y *= decay
            step_dy = self.scroll_velocity_y * dt * 25.0
            self.scroll_accumulator += step_dy
            momentum_dy = 0
            if abs(self.scroll_accumulator) >= 1.0:
                momentum_dy = int(self.scroll_accumulator)
                self.scroll_accumulator -= momentum_dy

            if abs(self.scroll_velocity_y) < self.MOMENTUM_MIN_VELOCITY:
                self.cancel_momentum_scroll()

            if momentum_dy != 0:
                return GestureState.SCROLLING, 0.0, momentum_dy, "Coasting Scroll...", screen_pos, None, None, None

        if not self.is_pinching:
            if pinch_intent:
                self.cancel_momentum_scroll()
                self.is_pinching = True
                self.pinch_start_time = now
                self.pinch_freeze_x = sx
                self.pinch_freeze_y = sy
                self.pinch_start_modifier = is_modifier_active
                self.is_dragging = False
                self.is_snipping = False
                self.snip_box = None
                current_action = GestureState.PINCHING
                status_msg = "Pinch Active"
                if is_modifier_active:
                    status_msg = "[SHIFT] Snip / Right Pinch Active"
                effective_pos = (self.pinch_freeze_x, self.pinch_freeze_y)
        else:
            # Currently pinching
            if index_thumb_min_dist > self.pinch_release_threshold:
                # Pinch released
                pinch_duration = now - self.pinch_start_time
                self.is_pinching = False

                if self.is_snipping:
                    # Snippet drag released -> trigger capture
                    self.snip_box = (self.pinch_freeze_x, self.pinch_freeze_y, sx, sy)
                    self.is_snipping = False
                    self.is_dragging = False
                    current_action = GestureState.SNIP_RELEASE
                    status_msg = "Snippet Captured"
                    effective_pos = (sx, sy)
                elif self.is_dragging:
                    self.is_dragging = False
                    current_action = GestureState.POINTING
                    status_msg = "Drag Released"
                    effective_pos = (sx, sy)
                else:
                    # Released within tap window (< 250ms)
                    if pinch_duration < self.tap_max_time:
                        # MODIFIER CHECK:
                        # If Left Hand was present during pinch -> RIGHT CLICK
                        if self.pinch_start_modifier or is_modifier_active:
                            current_action = GestureState.RIGHT_CLICK
                            status_msg = "[SHIFT] Right Click"
                        else:
                            # Left Hand down -> LEFT CLICK / DOUBLE CLICK
                            time_since_last_tap = now - self.last_tap_time
                            tap_dist_from_last = math.hypot(self.pinch_freeze_x - self.last_tap_x, self.pinch_freeze_y - self.last_tap_y)

                            if time_since_last_tap < self.double_click_window and tap_dist_from_last < 45.0:
                                current_action = GestureState.DOUBLE_CLICK
                                status_msg = "Double Click"
                                self.last_tap_time = 0.0
                            else:
                                current_action = GestureState.CLICK
                                status_msg = "Left Click"
                                self.last_tap_time = now
                                self.last_tap_x = self.pinch_freeze_x
                                self.last_tap_y = self.pinch_freeze_y
                    else:
                        # Long hold released without meeting drag distance -> Left Click at target
                        current_action = GestureState.CLICK
                        status_msg = "Left Click"

                    effective_pos = (self.pinch_freeze_x, self.pinch_freeze_y)
            else:
                # Pinch held
                pinch_duration = now - self.pinch_start_time
                dist_moved = math.hypot(sx - self.pinch_freeze_x, sy - self.pinch_freeze_y)

                if self.pinch_start_modifier or is_modifier_active:
                    # Modifier active -> Snipping Mode
                    if (dist_moved > 15.0 or pinch_duration > 0.20):
                        self.is_snipping = True
                        self.snip_box = (self.pinch_freeze_x, self.pinch_freeze_y, sx, sy)
                        current_action = GestureState.SNIP_DRAG
                        status_msg = "[SHIFT] Snip Area Selection"
                        effective_pos = (sx, sy)
                    else:
                        current_action = GestureState.PINCHING
                        status_msg = "[SHIFT] Right Pinch Locked"
                        effective_pos = (self.pinch_freeze_x, self.pinch_freeze_y)
                else:
                    # Normal mode -> Desktop Dragging
                    if (pinch_duration > self.drag_time_threshold and dist_moved > self.drag_dist_threshold) and not self.is_dragging:
                        self.is_dragging = True

                    if self.is_dragging:
                        current_action = GestureState.DRAGGING
                        status_msg = "Dragging"
                        effective_pos = (sx, sy)
                    else:
                        current_action = GestureState.PINCHING
                        status_msg = "Pinch Locked"
                        effective_pos = (self.pinch_freeze_x, self.pinch_freeze_y)

        return current_action, pinch_progress, 0, status_msg, effective_pos, None, None, self.snip_box

    def reset_pinch_states(self):
        self.is_pinching = False
        self.is_dragging = False
        self.is_snipping = False
        self.snip_box = None

    def reset_right_hand_states(self):
        self.reset_pinch_states()
        self.cancel_momentum_scroll()
        self.prev_scroll_y = None
        self.scroll_accumulator = 0.0
        self.prev_knuckle_pos = None
        self.prev_tip_pos = None
        self.prev_time = None
        self.three_finger_history.clear()

    def reset_transient_states(self):
        self.reset_right_hand_states()
        self.fist_start_time = None
        self.is_fist_active = False
        self.dual_fist_start_time = None
        self.is_dual_fist_active = False
        self.dual_open_start_time = None
        self.is_radial_active = False



