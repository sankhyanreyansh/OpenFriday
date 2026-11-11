"""
Unit tests for VoiceTranscriber audio capture, microphone resolution, and clipboard management.
"""

import time
import numpy as np
import pytest

from src.audio.voice_transcriber import (
    VoiceTranscriber,
    resolve_mic_device,
    paste_text_and_restore_clipboard,
)

try:
    from AppKit import NSPasteboard, NSStringPboardType
    COCOA_AVAILABLE = True
except ImportError:
    COCOA_AVAILABLE = False
    import pyperclip


def test_resolve_mic_device():
    """Verify microphone device resolution for default and explicit configurations."""
    # None and 'default' map to None (active system default)
    assert resolve_mic_device(None) is None
    assert resolve_mic_device("default") is None
    assert resolve_mic_device("") is None

    # Explicit integer index
    assert resolve_mic_device(2) == 2


def test_paste_text_and_restore_clipboard():
    """Verify text is pasted with a trailing space and original clipboard content is restored."""
    # 1. Setup initial clipboard content
    orig_text = "ORIGINAL_CLIPBOARD_TEST_CONTENT"
    if COCOA_AVAILABLE:
        pb = NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(orig_text, NSStringPboardType)
    else:
        pyperclip.copy(orig_text)

    # 2. Paste new transcription
    paste_text_and_restore_clipboard("hello world", restore_delay=0.08)

    # Immediately after paste call, clipboard should contain the transcribed text + trailing space
    if COCOA_AVAILABLE:
        pasted = pb.stringForType_(NSStringPboardType)
    else:
        pasted = pyperclip.paste()

    assert pasted == "hello world ", f"Expected 'hello world ', got '{pasted}'"

    # 3. Wait for the asynchronous background restoration thread (0.08s + margin)
    time.sleep(0.15)

    if COCOA_AVAILABLE:
        restored = pb.stringForType_(NSStringPboardType)
    else:
        restored = pyperclip.paste()

    assert restored == orig_text, f"Clipboard failed to restore! Got '{restored}'"


def test_voice_transcriber_discard_short_audio():
    """Verify audio recordings under 250ms are safely discarded without error."""
    vt = VoiceTranscriber(model_name="base.en")
    vt._is_recording = True

    # Under 250ms at 16kHz is < 4000 samples (e.g. 1000 samples = ~62ms)
    vt._audio_chunks = [np.zeros(1000, dtype=np.float32)]
    result = vt.stop_recording()
    assert result is None, "Expected brief audio (<250ms) to be discarded"

