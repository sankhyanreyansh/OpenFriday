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


class GestureRecognizer:
    """Analyzes 3D MediaPipe landmarks with robust two-finger scroll and dual-hand modifier mechanics."""

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

        self.last_tap_time: float = 0.0
        self.last_tap_x: float = 0.0
        self.last_tap_y: float = 0.0

        # Kinematic velocity tracking
        self.prev_knuckle_pos: Optional[Tuple[float, float]] = None
        self.prev_tip_pos: Optional[Tuple[float, float]] = None
        self.prev_time: Optional[float] = None

        # Two-finger scroll tracking
        self.prev_scroll_y: Optional[float] = None
        self.scroll_accumulator: float = 0.0
        self.scroll_sensitivity = 1.0

        # Three-finger space navigation & Mission Control tracking
        self.three_finger_history: List[Tuple[float, float, float]] = []
        self.last_swipe_time: float = 0.0
        self.swipe_cooldown: float = 0.60

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

    def _is_finger_curled(self, landmarks, tip_idx: int, pip_idx: int) -> bool:
        """Returns True if the finger tip is curled close to palm/wrist."""
        wrist = landmarks[self.WRIST]
        tip_dist = self._euclidean_dist(landmarks[tip_idx], wrist)
        pip_dist = self._euclidean_dist(landmarks[pip_idx], wrist)
        return tip_dist <= pip_dist * 1.15

    def get_stable_pointer_coords(self, landmarks) -> Tuple[float, float]:
        """
        Drives cursor navigation directly from Index MCP (Landmark 5, base knuckle).
        Does NOT move when fingers flex or tap during clicks.
        """
        mcp5 = landmarks[self.INDEX_MCP]
        return mcp5.x, mcp5.y

    def process(
        self,
        landmarks,
        screen_pos: Tuple[float, float],
        is_left_hand_present: bool = False,
        now: Optional[float] = None,
    ) -> Tuple[GestureState, float, int, str, Tuple[float, float], Optional[str]]:
        """
        Processes dominant hand landmarks, screen position, and modifier presence with strict precedence gating.
        Returns:
            (gesture_state, pinch_progress, scroll_dy, status_message, effective_screen_pos, nav_action)
        """
        if now is None:
            now = time.time()

        if landmarks is None or len(landmarks) < 21:
            self.reset_transient_states()
            return GestureState.NONE, 0.0, 0, "No Hand Detected", screen_pos, None

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
        is_index_extended = (
            (landmarks[self.INDEX_TIP].y < landmarks[self.INDEX_PIP].y) or
            (self._euclidean_dist(landmarks[self.INDEX_TIP], wrist) > self._euclidean_dist(landmarks[self.INDEX_PIP], wrist) * 1.08)
        )
        is_middle_extended = (
            (landmarks[self.MIDDLE_TIP].y < landmarks[self.MIDDLE_PIP].y) or
            (self._euclidean_dist(landmarks[self.MIDDLE_TIP], wrist) > self._euclidean_dist(landmarks[self.MIDDLE_PIP], wrist) * 1.08)
        )
        is_ring_extended = (
            (landmarks[self.RING_TIP].y < landmarks[self.RING_PIP].y) or
            (self._euclidean_dist(landmarks[self.RING_TIP], wrist) > self._euclidean_dist(landmarks[self.RING_PIP], wrist) * 1.08)
        )
        is_pinky_curled = (
            self._is_finger_curled(landmarks, self.PINKY_TIP, self.PINKY_PIP) or
            (landmarks[self.PINKY_TIP].y > landmarks[self.PINKY_PIP].y) or
            (self._euclidean_dist(landmarks[self.PINKY_TIP], wrist) <= self._euclidean_dist(landmarks[self.PINKY_PIP], wrist) * 1.15)
        )

        # =========================================================================
        # STEP 1: STRICT 3-FINGER SPATIAL NAVIGATION (Index + Middle + Ring Extended, Pinky Curled)
        # =========================================================================
        is_three_finger = is_index_extended and is_middle_extended and is_ring_extended and is_pinky_curled

        if is_three_finger:
            self.reset_pinch_states()
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
                # If |ΔX| > 0.08 and |ΔX| > 1.4 * |ΔY|
                if abs_dx > 0.08 and abs_dx > 1.4 * abs_dy:
                    if dx < 0:
                        # Hand moved Left -> trigger switch_space_right() (moves to Right space)
                        nav_action = "SPACE_RIGHT"
                        status_msg = "Swipe Left ➔ Space Right"
                    else:
                        # Hand moved Right -> trigger switch_space_left() (moves to Left space)
                        nav_action = "SPACE_LEFT"
                        status_msg = "Swipe Right ➔ Space Left"
                    print(f"[SWIPE DETECTED] Triggering action: {nav_action} (dx={dx:.3f}, dy={dy:.3f})")
                    self.last_swipe_time = now
                    self.three_finger_history.clear()

                # 2. Vertical Swipe Up Check (Mission Control)
                # If ΔY < -0.07 (upward in image coordinates) and |ΔY| > 1.3 * |ΔX|
                elif dy < -0.07 and abs_dy > 1.3 * abs_dx:
                    nav_action = "MISSION_CONTROL"
                    status_msg = "Swipe Up ➔ Mission Control"
                    print(f"[SWIPE DETECTED] Triggering action: {nav_action} (dx={dx:.3f}, dy={dy:.3f})")
                    self.last_swipe_time = now
                    self.three_finger_history.clear()

            return GestureState.SWIPE_NAV, 0.0, 0, status_msg, screen_pos, nav_action

        self.three_finger_history.clear()

        # =========================================================================
        # STEP 2: STRICT TWO-FINGER SCROLL CHECK (Index + Middle Extended, Ring Curled)
        # =========================================================================
        is_two_finger = is_index_extended and is_middle_extended

        if is_two_finger:
            # Force mode to SCROLL ONLY, completely bypassing all pinch/click detection
            self.reset_pinch_states()

            if is_left_hand_present:
                curr_y = (landmarks[self.INDEX_TIP].y + landmarks[self.MIDDLE_TIP].y) * 0.5
                scroll_dy = 0

                if self.prev_scroll_y is not None:
                    # Moving hand UP (curr_y < prev_scroll_y) -> delta_y > 0 (Scroll UP)
                    # Moving hand DOWN (curr_y > prev_scroll_y) -> delta_y < 0 (Scroll DOWN)
                    delta_y = (self.prev_scroll_y - curr_y) * 140.0 * self.scroll_sensitivity
                    self.scroll_accumulator += delta_y
                    if abs(self.scroll_accumulator) >= 1.0:
                        scroll_dy = int(self.scroll_accumulator)
                        self.scroll_accumulator -= scroll_dy
                self.prev_scroll_y = curr_y

                if scroll_dy > 0:
                    status_msg = f"Scrolling Up (+{scroll_dy})"
                elif scroll_dy < 0:
                    status_msg = f"Scrolling Down ({scroll_dy})"
                else:
                    status_msg = "[SHIFT] Scroll Active (Move Up/Down)"

                return GestureState.SCROLLING, 0.0, scroll_dy, status_msg, screen_pos, None
            else:
                self.prev_scroll_y = None
                self.scroll_accumulator = 0.0
                return GestureState.POINTING, 0.0, 0, "Two-Finger Neutral (Raise Left Hand to Scroll)", screen_pos, None

        self.prev_scroll_y = None
        self.scroll_accumulator = 0.0

        # =========================================================================
        # STEP 3: PINCH & CLICK DETECTION (Requires Middle Finger Curled)
        # =========================================================================
        middle_curled = self._is_finger_curled(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)

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
        if is_left_hand_present:
            status_msg = "[SHIFT] Modifier Active"
        effective_pos = (sx, sy)

        # Pinch intent requires middle finger to be curled
        pinch_intent = middle_curled and (
            index_thumb_min_dist < self.pinch_start_threshold or
            (is_stationary_knuckle and index_thumb_min_dist < (self.pinch_start_threshold * 1.15))
        )

        if not self.is_pinching:
            if pinch_intent:
                self.is_pinching = True
                self.pinch_start_time = now
                self.pinch_freeze_x = sx
                self.pinch_freeze_y = sy
                self.pinch_start_modifier = is_left_hand_present
                self.is_dragging = False
                current_action = GestureState.PINCHING
                status_msg = "Pinch Active"
                if is_left_hand_present:
                    status_msg = "[SHIFT] Right Pinch Active"
                effective_pos = (self.pinch_freeze_x, self.pinch_freeze_y)
        else:
            # Currently pinching
            if index_thumb_min_dist > self.pinch_release_threshold:
                # Pinch released
                pinch_duration = now - self.pinch_start_time
                self.is_pinching = False

                if self.is_dragging:
                    self.is_dragging = False
                    current_action = GestureState.POINTING
                    status_msg = "Drag Released"
                    effective_pos = (sx, sy)
                else:
                    # Released within tap window (< 250ms)
                    if pinch_duration < self.tap_max_time:
                        # MODIFIER CHECK:
                        # If Left Hand was present during pinch -> RIGHT CLICK
                        if self.pinch_start_modifier or is_left_hand_present:
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
                        # Long hold released without meeting 35px drag distance -> Left Click at target
                        current_action = GestureState.CLICK
                        status_msg = "Left Click"

                    effective_pos = (self.pinch_freeze_x, self.pinch_freeze_y)
            else:
                # Pinch held
                pinch_duration = now - self.pinch_start_time
                dist_moved = math.hypot(sx - self.pinch_freeze_x, sy - self.pinch_freeze_y)

                # Mandatory Spatial Movement Gate (Only in Normal Mode without modifier)
                if not self.pinch_start_modifier and not is_left_hand_present:
                    if (pinch_duration > self.drag_time_threshold and dist_moved > self.drag_dist_threshold) and not self.is_dragging:
                        self.is_dragging = True

                if self.is_dragging:
                    current_action = GestureState.DRAGGING
                    status_msg = "Dragging"
                    effective_pos = (sx, sy)
                else:
                    # Hold locked position during click/hold
                    current_action = GestureState.PINCHING
                    status_msg = "Pinch Locked"
                    if is_left_hand_present or self.pinch_start_modifier:
                        status_msg = "[SHIFT] Right Pinch Locked"
                    effective_pos = (self.pinch_freeze_x, self.pinch_freeze_y)

        return current_action, pinch_progress, 0, status_msg, effective_pos, None

    def reset_pinch_states(self):
        self.is_pinching = False
        self.is_dragging = False

    def reset_transient_states(self):
        self.reset_pinch_states()
        self.prev_scroll_y = None
        self.scroll_accumulator = 0.0
        self.prev_knuckle_pos = None
        self.prev_tip_pos = None
        self.prev_time = None
        self.three_finger_history.clear()

