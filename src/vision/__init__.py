"""Vision package for Open FRIDAY: Camera acquisition, hand tracking, and gesture recognition."""

from .vision_engine import VisionEngine, CameraGrabber, ensure_model_file
from .gesture_recognizer import (
    GestureRecognizer,
    GestureState,
    GestureData,
)
from .landmark_smoother import LandmarkSmoother
from .one_euro_filter import (
    OneEuroFilter,
    CursorSmoother,
    smoothing_factor,
    exponential_smoothing,
)

__all__ = [
    "VisionEngine",
    "CameraGrabber",
    "ensure_model_file",
    "GestureRecognizer",
    "GestureState",
    "GestureData",
    "LandmarkSmoother",
    "OneEuroFilter",
    "CursorSmoother",
    "smoothing_factor",
    "exponential_smoothing",
]
