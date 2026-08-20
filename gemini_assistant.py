"""
Local AI Assistant Engine for FRIDAY powered by Ollama (qwen2.5:0.5b) and native macOS Text-to-Speech.
Provides 100% offline, zero-latency local speech processing without API keys or external cloud dependencies.
"""

import subprocess
import threading
from typing import Optional, Callable
import ollama


class GeminiAssistant:
    """Lightweight local AI assistant client integrating Ollama (qwen2.5:0.5b) and macOS native TTS."""

    def __init__(self, model: str = "qwen2.5:0.5b", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host
        self.client = ollama.Client(host=self.host)
        self.system_instruction = (
            "You are FRIDAY, an ultra-fast, direct macOS AI assistant. "
            "Respond naturally in 1 to 2 concise sentences suitable for spoken audio. "
            "Never use markdown formatting, asterisks, bullet points, emojis, or code blocks."
        )

    def query(self, prompt: str, on_status_change: Optional[Callable[[str], None]] = None):
        """Sends prompt to local Ollama in a background thread and speaks the response."""
        if not prompt or not prompt.strip():
            return

        threading.Thread(
            target=self._process_query,
            args=(prompt.strip(), on_status_change),
            daemon=True,
        ).start()

    def _process_query(self, prompt: str, on_status_change: Optional[Callable[[str], None]]):
        try:
            if on_status_change:
                on_status_change("THINKING")

            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.6,
                    "num_predict": 100,
                },
            )

            # Safely extract reply content from dict or object
            if hasattr(response, "message") and hasattr(response.message, "content"):
                reply = response.message.content.strip()
            elif isinstance(response, dict) and "message" in response and "content" in response["message"]:
                reply = response["message"]["content"].strip()
            else:
                reply = str(response).strip()

            if not reply:
                reply = "I didn't receive a response."

            if on_status_change:
                on_status_change("SPEAKING")

            self._speak(reply)

            if on_status_change:
                on_status_change("IDLE")

        except Exception as e:
            print(f"[ERROR] Ollama request failed: {e}")
            if on_status_change:
                on_status_change("SPEAKING")
            self._speak("Sorry, I had trouble connecting to the local model.")
            if on_status_change:
                on_status_change("IDLE")

    def _speak(self, text: str):
        """Zero-latency local speech synthesis via native macOS say command."""
        if not text:
            return
        try:
            clean_text = text.replace('"', '\\"').replace("'", "’")
            subprocess.run(["say", clean_text])
        except Exception as e:
            print(f"[ERROR] macOS TTS error: {e}")


# Backward-compatible alias
AIAssistant = GeminiAssistant
