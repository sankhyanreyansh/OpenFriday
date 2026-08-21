"""
High-Performance Vision Engine with Multi-Hand Detection, Centroid Persistence & Dual-Hand Modifier.
Implements asynchronous CameraGrabber with single-slot frame buffering,
MediaPipe Hand Landmarker with num_hands=2, Spatial Hysteresis Centroid Tracking,
Velocity-Adaptive EMA jitter filtering, Active Interaction ROI, and native macOS Quartz event dispatch.
"""

import sys
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import math
import threading
import urllib.request
import re
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2
import mediapipe as mp

from PyQt6.QtCore import QThread, pyqtSignal, QObject, QBuffer, QIODevice
from PyQt6.QtGui import QImage, QGuiApplication
from landmark_smoother import LandmarkSmoother
from one_euro_filter import VelocityAdaptiveEMAFilter
from gesture_recognizer import GestureRecognizer, GestureState, GestureData
from mouse_controller import MouseController
from dictation_engine import VoiceDictationEngine
from ai_assistant import AIAssistant

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


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
        self.running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None

    def start_capture(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False

        self.running = True
        self.start()
        return True

    def run(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.005)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()



class VisionEngine(QThread):
    """Asynchronous camera capture and dual-hand gesture processing engine with centroid tracking."""

    # Signals for UI communication
    frame_ready = pyqtSignal(np.ndarray)
    gesture_updated = pyqtSignal(GestureData)
    fps_updated = pyqtSignal(float)
    status_changed = pyqtSignal(str)
    transcription_completed = pyqtSignal(str)
    ai_status_changed = pyqtSignal(str)
    context_image_captured = pyqtSignal(bytes)
    ai_response_generated = pyqtSignal(str)
    annotations_generated = pyqtSignal(list)

    def __init__(
        self,
        camera_id: int = 0,
        mouse_controller: Optional[MouseController] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.camera_id = camera_id
        self.mouse_controller = mouse_controller or MouseController()
        self.landmark_smoother = LandmarkSmoother(base_alpha=0.45, speed_coeff=8.0)
        self.gesture_recognizer = GestureRecognizer()
        
        # Local Voice Dictation Engine (faster-whisper) & Local Ollama AI Assistant
        self.dictation_engine = VoiceDictationEngine(model_size="base.en")
        self.ai_assistant = AIAssistant()
        self.ai_assistant.on_annotations_generated = lambda anns: self.annotations_generated.emit(anns)
        self.ai_assistant.signals.annotations_ready.connect(
            lambda anns, spoken: self.annotations_generated.emit(anns)
        )
        self.is_dictating = False
        self.current_dictation_mode: str = "CLIPBOARD"
        
        # Hardcoded Optimal Tracking Constants
        # smoothing = 0.2, responsiveness = 0.003, pinch_sensitivity = 0.4, scroll_speed = 1.0
        self.pointer_smoothing = 0.2
        self.pointer_beta = 0.003
        self.filter = VelocityAdaptiveEMAFilter(base_alpha=0.2, velocity_scale=0.003, deadzone_px=4.0)

        # Operational Flags
        self.running = False
        self.tracking_enabled = True
        self.mouse_control_enabled = True
        self.mirror_horizontal = True

        # Active Interaction Region (ROI: Inset 18% from webcam borders)
        self.roi_x_min = 0.18
        self.roi_x_max = 0.82
        self.roi_y_min = 0.18
        self.roi_y_max = 0.82

        # Screen dimensions
        self.screen_width, self.screen_height = self.mouse_controller.get_screen_size()
        self.gesture_recognizer.set_screen_dimensions(self.screen_width, self.screen_height)

        # Previous state memory & Centroid Persistence
        self.prev_state = GestureState.NONE
        self.was_dragging = False
        self.prev_dominant_centroid: Optional[Tuple[float, float]] = None
        self.prev_modifier_centroid: Optional[Tuple[float, float]] = None
        self.frame_counter = 0

        # Model path
        self.model_path = ensure_model_file()
        self.grabber: Optional[CameraGrabber] = None


    def update_settings(self, settings: dict):
        """Dynamically updates vision and control parameters from GUI settings."""
        if "camera_id" in settings and settings["camera_id"] != self.camera_id:
            self.camera_id = int(settings["camera_id"])
            if self.grabber:
                self.grabber.stop()
                self.grabber = CameraGrabber(self.camera_id)
                self.grabber.start_capture()
        if "tracking_enabled" in settings:
            self.tracking_enabled = bool(settings["tracking_enabled"])
        if "mouse_control_enabled" in settings:
            self.mouse_control_enabled = bool(settings["mouse_control_enabled"])
        if "mirror_horizontal" in settings:
            self.mirror_horizontal = bool(settings["mirror_horizontal"])
        if "margin_x" in settings:
            margin = float(settings["margin_x"])
            self.roi_x_min = margin
            self.roi_x_max = 1.0 - margin
        if "margin_y" in settings:
            margin = float(settings["margin_y"])
            self.roi_y_min = margin
            self.roi_y_max = 1.0 - margin
        if "smoothing" in settings:
            cutoff = float(settings["smoothing"])
            self.pointer_smoothing = cutoff
            beta = self.pointer_beta
            self.filter.update_params(cutoff, beta)
        if "responsiveness" in settings:
            beta = float(settings["responsiveness"])
            cutoff = self.pointer_smoothing
            self.pointer_beta = beta
            self.filter.update_params(cutoff, beta)
        if "pinch_threshold" in settings:
            self.gesture_recognizer.pinch_start_threshold = float(settings["pinch_threshold"])
            self.gesture_recognizer.pinch_release_threshold = float(settings["pinch_threshold"]) + 0.14
        if "scroll_sensitivity" in settings:
            self.gesture_recognizer.scroll_sensitivity = float(settings["scroll_sensitivity"])

    def dismiss_radial_menu(self):
        """Dismisses the radial menu modal state."""
        self.gesture_recognizer.dismiss_radial_menu()

    def stop(self):
        """Stops the vision thread safely."""
        self.running = False
        if self.is_dictating:
            self.is_dictating = False
            self.dictation_engine.stop_and_transcribe()
        if self.grabber:
            self.grabber.stop()
        self.wait(2000)


    def _map_to_screen(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """
        Maps normalized camera coordinates within Active Interaction ROI to full screen resolution
        using linear interpolation and boundary clamping (np.interp / np.clip).
        """
        sx = float(np.interp(norm_x, [self.roi_x_min, self.roi_x_max], [0, self.screen_width]))
        sy = float(np.interp(norm_y, [self.roi_y_min, self.roi_y_max], [0, self.screen_height]))
        sx = float(np.clip(sx, 0, self.screen_width - 1))
        sy = float(np.clip(sy, 0, self.screen_height - 1))
        return sx, sy

    def _init_landmarker(self):
        """Initializes MediaPipe HandLandmarker with explicit CPU delegate for zero Metal crashes."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _compute_palm_centroid(self, hand_landmarks) -> Tuple[float, float]:
        """Computes stable palm centroid from Wrist(0), IndexMCP(5), MiddleMCP(9), PinkyMCP(17)."""
        cx = (hand_landmarks[0].x + hand_landmarks[5].x + hand_landmarks[9].x + hand_landmarks[17].x) * 0.25
        cy = (hand_landmarks[0].y + hand_landmarks[5].y + hand_landmarks[9].y + hand_landmarks[17].y) * 0.25
        return cx, cy

    def _classify_hands(self, hands_landmarks, handedness) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Classifies detected hands into Right Hand (Dominant) and Left Hand (Modifier/PTT)
        using Spatial Hysteresis & Centroid Tracking.
        Returns:
            (right_hand_landmarks, left_hand_landmarks)
        """
        hands_count = len(hands_landmarks)
        if hands_count == 0:
            self.prev_dominant_centroid = None
            self.prev_modifier_centroid = None
            return None, None

        # MediaPipe handedness in mirrored camera frame:
        # Category "Left" in mirrored image corresponds to user's Physical Right Hand.
        # Category "Right" in mirrored image corresponds to user's Physical Left Hand.
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
                    elif label == target_right_label:
                        is_left = False

            if is_left:
                self.prev_modifier_centroid = centroids[0]
                self.prev_dominant_centroid = None
                return None, hands_landmarks[0]
            else:
                self.prev_dominant_centroid = centroids[0]
                self.prev_modifier_centroid = None
                return hands_landmarks[0], None

        # Two or more hands in view:
        # Check persistent centroids if available
        if self.prev_dominant_centroid is not None and self.prev_modifier_centroid is not None:
            d0_right = math.hypot(centroids[0][0] - self.prev_dominant_centroid[0], centroids[0][1] - self.prev_dominant_centroid[1])
            d1_right = math.hypot(centroids[1][0] - self.prev_dominant_centroid[0], centroids[1][1] - self.prev_dominant_centroid[1])

            if d0_right < d1_right:
                right_idx = 0
                left_idx = 1
            else:
                right_idx = 1
                left_idx = 0
        else:
            right_idx = None
            left_idx = None
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
                # Fallback: In mirrored frame, user's physical right hand is on right side (greater X)
                if centroids[0][0] > centroids[1][0]:
                    right_idx = 0
                    left_idx = 1
                else:
                    right_idx = 1
                    left_idx = 0

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

    def _on_ai_status_change(self, status: str):
        """Callback when AI Assistant changes status (THINKING, SPEAKING, CONTROLLING, IDLE)."""
        self.ai_status_changed.emit(status)

    def _on_ai_reply_generated(self, reply: str):
        """Callback when AI Assistant generates final text response."""
        self.ai_response_generated.emit(reply)

    def _on_unified_transcription_complete(self, text: str):
        """
        Unified transcript intent classification router:
        Checks if speech begins with or contains 'friday' wake-word.
        If 'Friday' detected:
            - Strip 'Friday' from utterance.
            - If query is empty, reply "I'm listening."
            - Else, send query to AIAssistant.
        If 'Friday' not detected:
            - Inject text directly into active search bar / text field via clipboard paste.
        """
        if not text or not text.strip():
            self.transcription_completed.emit("")
            return

        raw_text = text.strip()
        self.transcription_completed.emit(raw_text)

        # Regex for "Friday" or "Hey Friday", e.g. "^(hey\s+)?friday[,\s:!-.]*(.*)$"
        pattern = r'^(?:hey\s+)?friday[,\s:!-.]*(.*)$'
        match = re.search(pattern, raw_text, re.IGNORECASE)

        if match:
            query_prompt = match.group(1).strip()
            print(f"[WAKE WORD] 'Friday' wake-word detected in utterance: '{raw_text}' -> Query: '{query_prompt}'")
            if not query_prompt:
                ack_msg = "I'm listening."
                self._on_ai_status_change("SPEAKING")
                self._on_ai_reply_generated(ack_msg)
                self.ai_assistant._speak(ack_msg)
                self._on_ai_status_change("IDLE")
            else:
                self.ai_assistant.query(
                    query_prompt,
                    on_status_change=self._on_ai_status_change,
                    on_reply_generated=self._on_ai_reply_generated,
                    on_annotations_generated=lambda anns: self.annotations_generated.emit(anns),
                )
        else:
            print(f"[DICTATION] Routing text to desktop dictation: '{raw_text}'")
            self.dictation_engine._inject_text(raw_text)

    def run(self):
        """Main vision processing loop with multi-hand classification and centroid persistence."""
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

        with landmarker:
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

                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                gesture_data = GestureData(
                    is_tracking=False,
                    fps=fps_smoothing,
                    status_message="Searching for hand...",
                    is_modifier_active=False,
                )

                if self.tracking_enabled:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect(mp_image)

                    right_hand_landmarks = None
                    left_hand_landmarks = None

                    if result.hand_landmarks and len(result.hand_landmarks) > 0:
                        right_hand_landmarks, left_hand_landmarks = self._classify_hands(
                            result.hand_landmarks, getattr(result, "handedness", None)
                        )

                    if right_hand_landmarks is not None:
                        # Apply Landmark Pre-Smoothing on Right Hand only
                        smoothed_landmarks = self.landmark_smoother.smooth(right_hand_landmarks)

                        # Extract stable pointer coordinates directly from Right Index MCP (Landmark 5)
                        raw_x, raw_y = self.gesture_recognizer.get_stable_pointer_coords(smoothed_landmarks)

                        unfiltered_sx, unfiltered_sy = self._map_to_screen(raw_x, raw_y)
                        filtered_sx, filtered_sy = self.filter.filter(unfiltered_sx, unfiltered_sy, now)

                        gesture_state, pinch_progress, scroll_dy, status_msg, effective_pos, nav_action, radial_sector, snip_box = self.gesture_recognizer.process(
                            smoothed_landmarks,
                            (filtered_sx, filtered_sy),
                            left_landmarks=left_hand_landmarks,
                            now=now,
                        )

                        screen_x, screen_y = effective_pos

                        if self.mouse_control_enabled:
                            self._handle_mouse_events(gesture_state, screen_x, screen_y, scroll_dy, nav_action)

                        gesture_data = GestureData(
                            raw_x=raw_x,
                            raw_y=raw_y,
                            screen_x=screen_x,
                            screen_y=screen_y,
                            state=gesture_state,
                            pinch_progress=pinch_progress,
                            scroll_dy=scroll_dy,
                            hand_landmarks=smoothed_landmarks,
                            is_tracking=True,
                            confidence=0.95,
                            fps=fps_smoothing,
                            status_message=status_msg,
                            is_modifier_active=(left_hand_landmarks is not None),
                            nav_action=nav_action,
                            radial_sector=radial_sector,
                            snip_box=snip_box,
                        )
                    else:
                        # Right hand not in frame:
                        self.landmark_smoother.reset()
                        self.filter.reset()
                        if self.was_dragging:
                            self.mouse_controller.mouse_up()
                            self.was_dragging = False

                        gesture_state, pinch_progress, scroll_dy, status_msg, effective_pos, nav_action, radial_sector, snip_box = self.gesture_recognizer.process(
                            None,
                            (0.0, 0.0),
                            left_landmarks=left_hand_landmarks,
                            now=now,
                        )

                        gesture_data = GestureData(
                            raw_x=0.0,
                            raw_y=0.0,
                            screen_x=0.0,
                            screen_y=0.0,
                            state=gesture_state,
                            pinch_progress=0.0,
                            scroll_dy=0,
                            hand_landmarks=None,
                            is_tracking=False,
                            confidence=0.0,
                            fps=fps_smoothing,
                            status_message=status_msg,
                            is_modifier_active=(left_hand_landmarks is not None),
                            nav_action=None,
                            radial_sector=radial_sector,
                            snip_box=snip_box,
                        )

                    # Handle Screen Snippet Capture on SNIP_RELEASE
                    if gesture_state == GestureState.SNIP_RELEASE and snip_box is not None:
                        x1, y1, x2, y2 = snip_box
                        rx = min(x1, x2)
                        ry = min(y1, y2)
                        rw = abs(x2 - x1)
                        rh = abs(y2 - y1)
                        if rw > 20 and rh > 20:
                            screen = QGuiApplication.primaryScreen()
                            if screen:
                                pix = screen.grabWindow(0, int(rx), int(ry), int(rw), int(rh))
                                if not pix.isNull():
                                    buf = QBuffer()
                                    buf.open(QIODevice.OpenModeFlag.WriteOnly)
                                    pix.save(buf, "PNG")
                                    img_bytes = bytes(buf.data())
                                    self.ai_assistant.set_context_image(img_bytes)
                                    self.context_image_captured.emit(img_bytes)

                    # Handle Push-to-Talk Voice Dictation & Wake-Word Routing
                    if gesture_state == GestureState.LISTENING:
                        if not self.is_dictating:
                            self.is_dictating = True
                            self.dictation_engine.start_recording()
                    else:
                        if self.is_dictating:
                            self.is_dictating = False
                            self.dictation_engine.stop_and_transcribe(
                                on_complete_callback=self._on_unified_transcription_complete,
                                inject_clipboard=False
                            )

                self.gesture_updated.emit(gesture_data)
                self.fps_updated.emit(fps_smoothing)


        if self.is_dictating:
            self.is_dictating = False
            self.dictation_engine.stop_and_transcribe()

        self.grabber.stop()
        self.mouse_controller.release_all()
        self.status_changed.emit("Tracking Stopped")

    def _handle_mouse_events(
        self,
        state: GestureState,
        screen_x: float,
        screen_y: float,
        scroll_dy: int,
        nav_action: Optional[str] = None,
    ):
        """Executes native mouse actions according to gesture state machine."""
        if state == GestureState.NONE:
            if self.was_dragging:
                self.mouse_controller.mouse_up()
                self.was_dragging = False
            return

        # Snipping Mode (Freeze mouse drag and clicks)
        if state in (GestureState.SNIP_DRAG, GestureState.SNIP_RELEASE):
            if self.was_dragging:
                self.mouse_controller.mouse_up()
                self.was_dragging = False
            return

        # Voice Dictation & AI Assistant (Freeze cursor movement and clicks)
        if state in (GestureState.LISTENING, GestureState.TRANSCRIBING, GestureState.AI_LISTENING):
            if self.was_dragging:
                self.mouse_controller.mouse_up()
                self.was_dragging = False
            return

        # Radial Shortcut Menu (Freeze standard mouse actions & execute triggered shortcut)
        if state == GestureState.RADIAL_MENU:
            if self.was_dragging:
                self.mouse_controller.mouse_up()
                self.was_dragging = False
            if nav_action in ("ENTER", "NEW_TAB", "CLOSE_TAB", "ESCAPE"):
                self.mouse_controller.trigger_shortcut(nav_action)
            return

        # 3-Finger Spatial Navigation (Freeze cursor movement and clicks)
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

        # Pointer movement
        if state in (GestureState.POINTING, GestureState.PINCHING, GestureState.DRAGGING, GestureState.SCROLLING):
            self.mouse_controller.move_to(screen_x, screen_y)

        if state == GestureState.DRAGGING:
            if not self.was_dragging:
                self.mouse_controller.mouse_down(screen_x, screen_y)
                self.was_dragging = True
        else:
            if self.was_dragging:
                self.mouse_controller.mouse_up(screen_x, screen_y)
                self.was_dragging = False

        if state == GestureState.CLICK:
            self.mouse_controller.click(screen_x, screen_y)

        elif state == GestureState.DOUBLE_CLICK:
            self.mouse_controller.double_click(screen_x, screen_y)

        elif state == GestureState.RIGHT_CLICK:
            self.mouse_controller.right_click(screen_x, screen_y)

        elif state == GestureState.SCROLLING and scroll_dy != 0:
            self.mouse_controller.scroll(scroll_dy, 0, screen_x, screen_y)


