"""
Voice Transcription & Speech-to-Text Module for Open FRIDAY.
Provides low-latency microphone audio streaming via sounddevice (CoreAudio),
local speech-to-text inference with faster-whisper, and instant pasting
via macOS NSPasteboard and synthetic Command+V with automatic clipboard restoration.
"""

import time
import logging
import threading
from typing import List, Optional, Any, Callable

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from PyQt6.QtCore import QThread, pyqtSignal

from src.control.keystrokes import post_keyboard_event, KEY_V
from src.control import permissions

try:
    import Quartz.CoreGraphics as CG
    QUARTZ_AVAILABLE = True
except ImportError:
    QUARTZ_AVAILABLE = False

try:
    from AppKit import NSPasteboard, NSStringPboardType
    COCOA_AVAILABLE = True
except ImportError:
    COCOA_AVAILABLE = False
    import pyperclip

logger = logging.getLogger("OpenFriday.VoiceTranscriber")


def resolve_mic_device(device_setting: Optional[Any] = None) -> Optional[int]:
    """
    Resolves the microphone device index from user configuration.
    If None or 'default', returns None (sounddevice uses the active system default input).
    If an int is provided, returns that int.
    If a string is provided, searches sounddevice device names for a matching substring.
    """
    if device_setting is None or device_setting == "default" or device_setting == "":
        return None
    if isinstance(device_setting, int):
        return device_setting
    if isinstance(device_setting, str):
        try:
            devices = sd.query_devices()
            setting_lower = device_setting.lower()
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0 and setting_lower in dev.get("name", "").lower():
                    return idx
        except Exception as e:
            logger.warning("Error querying audio devices for mic '%s': %s", device_setting, e)
    return None


def paste_text_and_restore_clipboard(text: str, restore_delay: float = 0.15):
    """
    Pastes text with a trailing space into the currently active macOS application via Command+V,
    and automatically restores the user's prior clipboard contents after a brief delay.
    """
    if not text:
        return

    formatted_text = text.strip() + " "
    orig_text: Optional[str] = None

    # 1. Save original clipboard and copy new transcription
    if COCOA_AVAILABLE:
        try:
            pb = NSPasteboard.generalPasteboard()
            orig_text = pb.stringForType_(NSStringPboardType)
            pb.clearContents()
            pb.setString_forType_(formatted_text, NSStringPboardType)
        except Exception as e:
            logger.warning("NSPasteboard copy error: %s", e)
    else:
        try:
            orig_text = pyperclip.paste()
            pyperclip.copy(formatted_text)
        except Exception as e:
            logger.warning("pyperclip copy error: %s", e)

    # 2. Dispatch native Command + V keystroke
    cmd_flag = CG.kCGEventFlagMaskCommand if QUARTZ_AVAILABLE else 0
    post_keyboard_event(KEY_V, cmd_flag)

    # 3. Restore previous clipboard content asynchronously
    def _restore():
        time.sleep(restore_delay)
        if COCOA_AVAILABLE:
            try:
                pb = NSPasteboard.generalPasteboard()
                pb.clearContents()
                if orig_text is not None:
                    pb.setString_forType_(orig_text, NSStringPboardType)
            except Exception as re:
                logger.debug("NSPasteboard restore error: %s", re)
        else:
            try:
                if orig_text is not None:
                    pyperclip.copy(orig_text)
            except Exception as re:
                logger.debug("pyperclip restore error: %s", re)

    threading.Thread(target=_restore, daemon=True, name="ClipboardRestore").start()


class TranscriptionWorker(QThread):
    """
    Asynchronous worker thread for local Whisper model transcription.
    Runs speech-to-text inference in the background so the 60 FPS vision engine loop never blocks.
    """

    transcription_completed = pyqtSignal(str)
    transcription_failed = pyqtSignal(str)

    def __init__(self, transcriber: "VoiceTranscriber", audio_data: np.ndarray):
        super().__init__()
        self.transcriber = transcriber
        self.audio_data = audio_data

    def run(self):
        try:
            text = self.transcriber.transcribe(self.audio_data)
            if text:
                paste_text_and_restore_clipboard(text)
                self.transcription_completed.emit(text)
            else:
                self.transcription_failed.emit("No speech detected.")
        except Exception as e:
            logger.exception("Voice transcription error: %s", e)
            self.transcription_failed.emit(str(e))


class VoiceTranscriber:
    """
    Voice Transcriber controller using faster-whisper and sounddevice.
    Handles continuous non-blocking microphone stream capture, model management,
    and background transcription worker dispatch.
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_name: str = "base.en",
        mic_device: Optional[Any] = None,
        language: str = "en",
    ):
        self.model_name = model_name
        self.mic_device = mic_device
        self.language = language

        self._model: Optional[WhisperModel] = None
        self._model_lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._audio_chunks: List[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._is_recording = False
        self._worker: Optional[TranscriptionWorker] = None

    def start_background_model_preload(self):
        """Asynchronously pre-loads the Whisper model into RAM at startup."""
        def _load():
            try:
                self._ensure_model_loaded()
            except Exception as e:
                logger.debug("Whisper model preload skipped or interrupted: %s", e)

        threading.Thread(target=_load, daemon=True, name="WhisperPreload").start()

    def _ensure_model_loaded(self) -> WhisperModel:
        with self._model_lock:
            if self._model is None:
                logger.info("Loading local Whisper model '%s' (CPU int8)...", self.model_name)
                t0 = time.time()
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                logger.info("Whisper model loaded in %.2fs", time.time() - t0)
            return self._model

    def _audio_callback(self, indata, frames, time_info, status):
        """Streaming callback from sounddevice; stores audio chunks thread-safely."""
        if status:
            logger.debug("sounddevice status: %s", status)
        with self._buffer_lock:
            if self._is_recording:
                self._audio_chunks.append(indata.copy())

    def start_recording(self) -> bool:
        """
        Starts streaming microphone audio non-blocking via sounddevice.
        Returns True on success, False if microphone is inaccessible.
        """
        if self._is_recording:
            return True

        with self._buffer_lock:
            self._audio_chunks.clear()
            self._is_recording = True

        dev_idx = resolve_mic_device(self.mic_device)
        try:
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=dev_idx,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("Voice recording started on mic device: %s", dev_idx if dev_idx is not None else "Active System Default")
            return True
        except Exception as e:
            logger.error("Failed to open microphone stream: %s. Opening macOS settings...", e)
            permissions.open_microphone_settings()
            with self._buffer_lock:
                self._is_recording = False
            return False

    def stop_recording(self) -> Optional[np.ndarray]:
        """
        Stops audio recording and returns the raw 16kHz float32 audio numpy array.
        Returns None if recording was too brief or empty.
        """
        with self._buffer_lock:
            if not self._is_recording:
                return None
            self._is_recording = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Error stopping audio stream: %s", e)
            finally:
                self._stream = None

        with self._buffer_lock:
            if not self._audio_chunks:
                return None
            audio = np.concatenate(self._audio_chunks, axis=0).flatten()
            self._audio_chunks.clear()

        # Discard recordings shorter than 250ms (4000 samples at 16kHz)
        if len(audio) < (self.SAMPLE_RATE * 0.25):
            logger.debug("Recorded audio was too brief (<250ms); ignoring.")
            return None

        return audio

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Runs local inference on the provided 16kHz float32 audio array.
        Returns combined transcribed text.
        """
        model = self._ensure_model_loaded()
        segments, info = model.transcribe(
            audio_data,
            beam_size=5,
            language=self.language,
            condition_on_previous_text=False,
        )
        text = " ".join([segment.text.strip() for segment in segments]).strip()
        logger.info("Transcribed speech: '%s'", text)
        return text

    def start_transcription_worker(
        self,
        audio_data: np.ndarray,
        on_completed: Optional[Callable[[str], None]] = None,
        on_failed: Optional[Callable[[str], None]] = None,
    ) -> TranscriptionWorker:
        """Launches background transcription worker thread with Qt signal connections."""
        worker = TranscriptionWorker(self, audio_data)
        if on_completed:
            worker.transcription_completed.connect(on_completed)
        if on_failed:
            worker.transcription_failed.connect(on_failed)
        self._worker = worker
        worker.start()
        return worker
