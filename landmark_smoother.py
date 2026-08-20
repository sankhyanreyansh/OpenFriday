"""
Temporal Landmark Pre-Smoothing Filter.
Filters raw MediaPipe 3D landmark coordinates before scale and gesture calculations,
eliminating high-frequency camera sensor noise and quantization jitter.
"""

from typing import List, Optional
from types import SimpleNamespace


class SmoothedLandmark:
    """Represents a smoothed 3D landmark with x, y, z coordinates."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"SmoothedLandmark(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


class LandmarkSmoother:
    """
    Applies adaptive exponential smoothing across all 21 hand landmarks.
    - Low-speed motion: Higher smoothing factor (alpha ~ 0.35) removes sensor noise.
    - High-speed motion: Lower smoothing factor (alpha ~ 0.85) ensures zero tracking latency.
    """

    def __init__(self, base_alpha: float = 0.45, speed_coeff: float = 8.0, num_landmarks: int = 21):
        self.base_alpha = float(base_alpha)
        self.speed_coeff = float(speed_coeff)
        self.num_landmarks = num_landmarks
        self.prev_landmarks: Optional[List[SmoothedLandmark]] = None

    def smooth(self, raw_landmarks) -> List[SmoothedLandmark]:
        """
        Smooths a list of 21 raw landmarks from MediaPipe.
        Returns a list of SmoothedLandmark instances.
        """
        if raw_landmarks is None or len(raw_landmarks) < self.num_landmarks:
            self.reset()
            return []

        if self.prev_landmarks is None or len(self.prev_landmarks) != len(raw_landmarks):
            # Initialize with raw landmarks
            self.prev_landmarks = [
                SmoothedLandmark(lm.x, lm.y, getattr(lm, "z", 0.0))
                for lm in raw_landmarks
            ]
            return self.prev_landmarks

        smoothed_list: List[SmoothedLandmark] = []

        for i, raw in enumerate(raw_landmarks):
            prev = self.prev_landmarks[i]
            raw_z = getattr(raw, "z", 0.0)

            # Compute motion distance since previous frame
            dx = raw.x - prev.x
            dy = raw.y - prev.y
            dz = raw_z - prev.z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5

            # Adaptive alpha: increases with motion speed
            alpha = min(1.0, max(self.base_alpha, self.base_alpha + dist * self.speed_coeff))

            sx = alpha * raw.x + (1.0 - alpha) * prev.x
            sy = alpha * raw.y + (1.0 - alpha) * prev.y
            sz = alpha * raw_z + (1.0 - alpha) * prev.z

            smoothed = SmoothedLandmark(sx, sy, sz)
            smoothed_list.append(smoothed)

        self.prev_landmarks = smoothed_list
        return smoothed_list

    def reset(self):
        """Resets smoothing history."""
        self.prev_landmarks = None
