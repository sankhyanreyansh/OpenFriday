"""
Local Voice Dictation Engine using faster-whisper on Apple Silicon.
Provides low-latency push-to-talk speech recognition and automatic clipboard injection via Cmd+V.
"""

import threading
import numpy as np
import sounddevice as sd
import pyperclip
import subprocess
from faster_whisper import WhisperModel


class VoiceDictationEngine:
    """Local Speech-to-Text Engine running faster-whisper on CPU with int8 quantization."""

    def __init__(self, model_size="base.en"):
        # Load model with int8 quantization for near-instant Apple Silicon inference
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.sample_rate = 16000
        self.audio_frames = []
        self.is_recording = False
        self.stream = None
        self.lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            with self.lock:
                self.audio_frames.append(indata.copy())

    def start_recording(self):
        with self.lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.audio_frames = []

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback
            )
            self.stream.start()
        except Exception as e:
            with self.lock:
                self.is_recording = False

    def stop_and_transcribe(self, on_complete_callback=None):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            frames_copy = list(self.audio_frames)
            self.audio_frames = []

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        # Run transcription in a background thread to prevent UI freezing
        threading.Thread(target=self._process_transcription, args=(frames_copy, on_complete_callback), daemon=True).start()

    def _process_transcription(self, frames, on_complete_callback):
        if not frames:
            if on_complete_callback:
                on_complete_callback("")
            return

        audio_data = np.concatenate(frames, axis=0).flatten()
        # Ignore accidental taps shorter than 0.35 seconds
        if len(audio_data) < self.sample_rate * 0.35:
            if on_complete_callback:
                on_complete_callback("")
            return

        try:
            segments, _ = self.model.transcribe(audio_data, beam_size=2, language="en")
            text = " ".join([seg.text for seg in segments]).strip()
        except Exception:
            text = ""

        if text:
            self._inject_text(text)

        if on_complete_callback:
            on_complete_callback(text)


    def _inject_text(self, text: str):
        """Injects text into active search bar / text field via clipboard paste."""
        try:
            pyperclip.copy(text)
            # Simulate Cmd + V via AppleScript for instant, accurate typing
            script = 'tell application "System Events" to keystroke "v" using command down'
            subprocess.run(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[DICTATION] Text injection error: {e}")
