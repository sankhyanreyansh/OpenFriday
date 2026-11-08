"""
Vision Engine for Open FRIDAY: 60 FPS multi-threaded camera acquisition,
MediaPipe Hand Landmarker tracking, kinematic cursor filtering, and gesture dispatch.
"""

import os
import time
import math
import urllib.request
import threading
from typing import Optional, Tuple, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from PyQt6.QtCore import QThread, pyqtSignal

from src.control.mouse_controller import MouseController
from src.vision.gesture_recognizer import (
    GestureRecognizer,
    GestureState,
    GestureData,
)
from src.vision.landmark_smoother import LandmarkSmoother
from src.vision.one_euro_filter import CursorSmoother

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_MODEL_PATH = os.path.join(ROOT_DIR, "hand_landmarker.task")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
DEFAULT_MODEL_PATH = ROOT_MODEL_PATH if os.path.exists(ROOT_MODEL_PATH) else LOCAL_MODEL_PATH


def ensure_model_file(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """Ensures hand landmarker task model file is present locally."""
    if not os.path.exists(model_path):
        print(f"Downloading hand landmarker model to {model_path}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print(f"Model downloaded successfully ({os.path.getsize(model_path)} bytes).")
        except Exception as e:
            print(f"Error downloading model: {e}")
    return model_path


class CameraGrabber(threading.Thread):
    """
    Dedicated camera capture thread.
    Continuously grabs frames from OpenCV VideoCapture and keeps only the latest frame
    to completely eliminate camera buffer latency and frame queue buildup.
    """

    def __init__(self, camera_id: int = 0):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False

    def start_capture(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_id}")
            return False

        # Request 60 FPS and 720p resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        self.start()
        return True

    def run(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.005)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None


class VisionEngine(QThread):
    """
    Core Computer Vision Worker Thread.
    Runs camera capture, MediaPipe HandLandmarker inference, landmark & cursor smoothing,
    and dispatches native mouse events via MouseController.
    """

    gesture_updated = pyqtSignal(GestureData)
    fps_updated = pyqtSignal(float)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        camera_id: int = 0,
        mouse_controller: Optional[MouseController] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.camera_id = camera_id
        self.mouse_controller = mouse_controller or MouseController()

        if model_path is None:
            self.model_path = ensure_model_file()
        else:
            self.model_path = model_path

        # Sub-modules
        self.gesture_recognizer = GestureRecognizer()
        self.landmark_smoother = LandmarkSmoother(alpha_slow=0.35, alpha_fast=0.88)
        self.cursor_smoother = CursorSmoother(min_cutoff=0.2, beta=0.003)

        # Operational Settings
        self.tracking_enabled = True
        self.mouse_control_enabled = True
        self.mirror_horizontal = True
        self.running = False

        # Screen Coordinate Mapping (with margin inset)
        self.screen_width = float(self.mouse_controller.screen_width)
        self.screen_height = float(self.mouse_controller.screen_height)
        self.gesture_recognizer.set_screen_dimensions(self.screen_width, self.screen_height)

        self.margin_x = 0.16
        self.margin_y = 0.16

        # State tracking
        self.was_dragging = False
        self.last_zoom_time: float = 0.0

        # Centroid persistence for dual-hand classification
        self.prev_dominant_centroid: Optional[Tuple[float, float]] = None
        self.prev_modifier_centroid: Optional[Tuple[float, float]] = None

    def update_settings(self, settings: dict):
        """Applies dynamic runtime configuration updates."""
        if "tracking_enabled" in settings:
            self.tracking_enabled = bool(settings["tracking_enabled"])
        if "mouse_control_enabled" in settings:
            self.mouse_control_enabled = bool(settings["mouse_control_enabled"])
        if "mirror_horizontal" in settings:
            self.mirror_horizontal = bool(settings["mirror_horizontal"])
        if "margin_x" in settings:
            self.margin_x = float(settings["margin_x"])
        if "margin_y" in settings:
            self.margin_y = float(settings["margin_y"])
        if "scroll_sensitivity" in settings:
            self.gesture_recognizer.scroll_sensitivity = float(settings["scroll_sensitivity"])

        min_cutoff = settings.get("min_cutoff", 0.2)
        beta = settings.get("beta", 0.003)
        self.cursor_smoother = CursorSmoother(min_cutoff=min_cutoff, beta=beta)

    def dismiss_radial_menu(self):
        """Manually dismisses the radial shortcut wheel."""
        self.gesture_recognizer.dismiss_radial_menu()

    def stop(self):
        """Signals vision loop to stop and waits for thread completion."""
        self.running = False
        self.wait(1500)

    def _map_to_screen(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """Maps normalized landmark coordinates [0, 1] to screen dimensions with margin inset."""
        clamped_x = (norm_x - self.margin_x) / max(1.0 - 2 * self.margin_x, 0.01)
        clamped_y = (norm_y - self.margin_y) / max(1.0 - 2 * self.margin_y, 0.01)
        sx = max(0.0, min(1.0, clamped_x)) * self.screen_width
        sy = max(0.0, min(1.0, clamped_y)) * self.screen_height
        return sx, sy

    def _init_landmarker(self) -> vision.HandLandmarker:
        """Initializes the MediaPipe HandLandmarker in IMAGE mode."""
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _compute_palm_centroid(self, hand_landmarks) -> Tuple[float, float]:
        """Computes stable palm centroid from Wrist(0), IndexMCP(5), MiddleMCP(9), PinkyMCP(17)."""
        hl = hand_landmarks
        def get_xy(p):
            return (p.x, p.y) if hasattr(p, 'x') else (p[0], p[1])
        p0, p5, p9, p17 = get_xy(hl[0]), get_xy(hl[5]), get_xy(hl[9]), get_xy(hl[17])
        cx = (p0[0] + p5[0] + p9[0] + p17[0]) * 0.25
        cy = (p0[1] + p5[1] + p9[1] + p17[1]) * 0.25
        return cx, cy

    def _classify_hands(self, hands_landmarks, handedness) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Classifies detected hands into Right Hand (Dominant) and Left Hand (Modifier).
        Returns: (right_hand_landmarks, left_hand_landmarks)
        """
        hands_count = len(hands_landmarks)
        if hands_count == 0:
            self.prev_dominant_centroid = None
            self.prev_modifier_centroid = None
            return None, None

        target_right_label = "Left" if self.mirror_horizontal else "Right"
        target_left_label = "Right" if self.mirror_horizontal else "Left"

        centroids = [self._compute_palm_centroid(hl) for hl in hands_landmarks]

        if hands_count == 1:
            is_left = False
            if handedness and len(handedness) >= 1:
                h_class = getattr(handedness[0], "classification", handedness[0])
                if h_class and len(h_class) > 0:
                    label = getattr(h_class[0], "label", getattr(h_class[0], "category_name", ""))
                    if label == target_left_label:
                        is_left = True

            if is_left:
                self.prev_modifier_centroid = centroids[0]
                self.prev_dominant_centroid = None
                return None, hands_landmarks[0]
            else:
                self.prev_dominant_centroid = centroids[0]
                self.prev_modifier_centroid = None
                return hands_landmarks[0], None

        # Two or more hands in view: match against persistent centroids if available
        if self.prev_dominant_centroid is not None and self.prev_modifier_centroid is not None:
            d0_right = math.hypot(centroids[0][0] - self.prev_dominant_centroid[0], centroids[0][1] - self.prev_dominant_centroid[1])
            d1_right = math.hypot(centroids[1][0] - self.prev_dominant_centroid[0], centroids[1][1] - self.prev_dominant_centroid[1])
            if d0_right < d1_right:
                right_idx, left_idx = 0, 1
            else:
                right_idx, left_idx = 1, 0
        else:
            right_idx, left_idx = None, None
            if handedness and len(handedness) >= 2:
                for idx, h_info in enumerate(handedness[:2]):
                    h_class = getattr(h_info, "classification", h_info)
                    if h_class and len(h_class) > 0:
                        label = getattr(h_class[0], "label", getattr(h_class[0], "category_name", ""))
                        if label == target_right_label and right_idx is None:
                            right_idx = idx
                        elif label == target_left_label and left_idx is None:
                            left_idx = idx

            if right_idx is None or left_idx is None or right_idx == left_idx:
                if centroids[0][0] > centroids[1][0]:
                    right_idx, left_idx = 0, 1
                else:
                    right_idx, left_idx = 1, 0

        # Update persistent centroids with EMA smoothing
        alpha = 0.65
        self.prev_dominant_centroid = (
            alpha * centroids[right_idx][0] + (1 - alpha) * (self.prev_dominant_centroid[0] if self.prev_dominant_centroid else centroids[right_idx][0]),
            alpha * centroids[right_idx][1] + (1 - alpha) * (self.prev_dominant_centroid[1] if self.prev_dominant_centroid else centroids[right_idx][1]),
        )
        self.prev_modifier_centroid = (
            alpha * centroids[left_idx][0] + (1 - alpha) * (self.prev_modifier_centroid[0] if self.prev_modifier_centroid else centroids[left_idx][0]),
            alpha * centroids[left_idx][1] + (1 - alpha) * (self.prev_modifier_centroid[1] if self.prev_modifier_centroid else centroids[left_idx][1]),
        )

        return hands_landmarks[right_idx], hands_landmarks[left_idx]

    def run(self):
        """Main vision loop: continuous camera acquisition, inference, and mouse event dispatch."""
        self.running = True
        self.status_changed.emit("Initializing Camera...")

        self.grabber = CameraGrabber(self.camera_id)
        if not self.grabber.start_capture():
            self.status_changed.emit("Failed to open camera. Check permissions.")
            self.running = False
            return

        self.status_changed.emit("Tracking Active")

        try:
            landmarker = self._init_landmarker()
        except Exception as e:
            print(f"Error creating HandLandmarker: {e}")
            self.status_changed.emit(f"Landmarker init failed: {e}")
            self.grabber.stop()
            self.running = False
            return

        prev_time = time.time()
        fps_smoothing = 0.0

        try:
            while self.running:
                frame = self.grabber.get_latest_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                now = time.time()
                dt = now - prev_time
                prev_time = now
                curr_fps = 1.0 / max(dt, 0.001)
                fps_smoothing = 0.9 * fps_smoothing + 0.1 * curr_fps if fps_smoothing > 0 else curr_fps

                if self.mirror_horizontal:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not self.tracking_enabled:
                    time.sleep(0.02)
                    continue

                # Inference Downsampling: 640x360 cuts inference CPU compute in half
                infer_frame = cv2.resize(rgb_frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_frame)
                result = landmarker.detect(mp_image)

                right_hand_landmarks = None
                left_hand_landmarks = None

                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    right_hand_landmarks, left_hand_landmarks = self._classify_hands(
                        result.hand_landmarks, getattr(result, "handedness", None)
                    )

                if right_hand_landmarks is not None:
                    # Apply Landmark Pre-Smoothing on Right Hand
                    smoothed_landmarks = self.landmark_smoother.smooth(right_hand_landmarks)

                    # Extract stable pointer coordinates from Right Index MCP (Landmark 5)
                    raw_x, raw_y = self.gesture_recognizer.get_stable_pointer_coords(smoothed_landmarks)
                    unfiltered_sx, unfiltered_sy = self._map_to_screen(raw_x, raw_y)

                    # Apply One-Euro dynamic smoothing to cursor coordinates
                    cursor_x, cursor_y = self.cursor_smoother.smooth(unfiltered_sx, unfiltered_sy)

                    # Check for Dual-Hand Pinch Spatial Zoom
                    zoom_delta = 0.0
                    if left_hand_landmarks is not None:
                        left_is_pinched = self.gesture_recognizer.is_hand_pinched(left_hand_landmarks)
                        right_is_pinched = self.gesture_recognizer.is_hand_pinched(smoothed_landmarks)
                        zoom_delta = self.gesture_recognizer.process_dual_hand_zoom(
                            left_hand_landmarks,
                            smoothed_landmarks,
                            left_is_pinched,
                            right_is_pinched,
                        )

                    if self.gesture_recognizer.dual_pinch_active:
                        if self.was_dragging:
                            self.mouse_controller.mouse_up()
                            self.was_dragging = False

                        zoom_status = "ZOOM"
                        if zoom_delta != 0.0 and self.mouse_control_enabled:
                            if now - self.last_zoom_time >= 0.06:
                                if zoom_delta > 0:
                                    self.mouse_controller.zoom("in", steps=1)
                                    zoom_status = "ZOOM IN"
                                else:
                                    self.mouse_controller.zoom("out", steps=1)
                                    zoom_status = "ZOOM OUT"
                                self.last_zoom_time = now
                            else:
                                zoom_status = "ZOOM IN" if zoom_delta > 0 else "ZOOM OUT"

                        gesture_data = GestureData(
                            raw_x=raw_x,
                            raw_y=raw_y,
                            screen_x=cursor_x,
                            screen_y=cursor_y,
                            state=GestureState.ZOOM,
                            pinch_progress=1.0,
                            hand_landmarks=smoothed_landmarks,
                            is_tracking=True,
                            fps=fps_smoothing,
                            status_message=zoom_status,
                            is_modifier_active=True,
                        )
                    else:
                        gesture_data = self.gesture_recognizer.process(
                            smoothed_landmarks,
                            (cursor_x, cursor_y),
                            left_landmarks=left_hand_landmarks,
                            now=now,
                        )
                        gesture_data.raw_x = raw_x
                        gesture_data.raw_y = raw_y
                        gesture_data.fps = fps_smoothing

                        if self.mouse_control_enabled:
                            self._handle_mouse_events(gesture_data)
                else:
                    self.landmark_smoother.reset()
                    self.cursor_smoother.reset()
                    if self.was_dragging:
                        self.mouse_controller.mouse_up()
                        self.was_dragging = False

                    gesture_data = self.gesture_recognizer.process(
                        None,
                        (0.0, 0.0),
                        left_landmarks=left_hand_landmarks,
                        now=now,
                    )
                    gesture_data.fps = fps_smoothing

                self.gesture_updated.emit(gesture_data)
                self.fps_updated.emit(fps_smoothing)

        except Exception as run_err:
            print(f"[VISION ENGINE ERROR] Run loop exception: {run_err}")
        finally:
            try:
                landmarker.close()
            except Exception:
                pass

            if hasattr(self, 'grabber') and self.grabber:
                self.grabber.stop()
            self.mouse_controller.release_all()
            self.status_changed.emit("Tracking Stopped")

    def _handle_mouse_events(self, data: GestureData):
        """Executes native mouse actions according to gesture state."""
        state = data.state
        screen_x = data.screen_x
        screen_y = data.screen_y
        scroll_dy = data.scroll_dy
        nav_action = data.nav_action

        if state == GestureState.NONE:
            if self.was_dragging:
                self.mouse_controller.mouse_up()
                self.was_dragging = False
            return

        # Continuous cursor movement across navigation
        self.mouse_controller.move_to(screen_x, screen_y)

        # Radial Shortcut Menu (Execute triggered shortcut)
        if state == GestureState.RADIAL_MENU:
            if self.was_dragging:
                self.mouse_controller.mouse_up()
                self.was_dragging = False
            if nav_action:
                self.mouse_controller.trigger_shortcut(nav_action)
            return

        # 3-Finger Spatial Navigation (Spaces / Mission Control)
        if state == GestureState.SWIPE_NAV:
            if self.was_dragging:
                self.mouse_controller.mouse_up()
                self.was_dragging = False
            if nav_action == "SPACE_RIGHT":
                self.mouse_controller.switch_space_right()
            elif nav_action == "SPACE_LEFT":
                self.mouse_controller.switch_space_left()
            elif nav_action == "MISSION_CONTROL":
                self.mouse_controller.trigger_mission_control()
            return

        # Dragging state
        if state == GestureState.DRAGGING:
            if not self.was_dragging:
                self.mouse_controller.mouse_down(screen_x, screen_y)
                self.was_dragging = True
        else:
            if self.was_dragging:
                self.mouse_controller.mouse_up(screen_x, screen_y)
                self.was_dragging = False

        # Click actions
        if state == GestureState.CLICK:
            self.mouse_controller.click(screen_x, screen_y)
        elif state == GestureState.DOUBLE_CLICK:
            self.mouse_controller.double_click(screen_x, screen_y)
        elif state == GestureState.RIGHT_CLICK:
            self.mouse_controller.right_click(screen_x, screen_y)
        elif state == GestureState.SCROLLING and scroll_dy != 0:
            self.mouse_controller.scroll(scroll_dy, 0, screen_x, screen_y)
