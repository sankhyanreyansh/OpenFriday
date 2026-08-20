"""
Local AI Assistant Engine for FRIDAY powered by Ollama and native macOS Text-to-Speech.
Provides 100% offline, zero-latency local speech and deterministic system tool execution (opening websites, launching apps).
"""

import base64
import subprocess
import threading
from typing import Optional, Callable, Dict, Any, List
import ollama

from system_tools import open_website, open_application


class AIAssistant:
    """Lightweight local AI assistant client integrating Ollama function calling, vision, and macOS native TTS."""

    def __init__(self, model: str = "qwen2.5:0.5b", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.vision_model = "qwen2.5vl:3b"
        self.host = host
        self.client = ollama.Client(host=self.host)
        self.system_instruction = (
            "You are FRIDAY, an ultra-fast, direct macOS AI assistant. "
            "Respond naturally in 1 to 2 concise sentences suitable for spoken audio. "
            "Never use markdown formatting, asterisks, bullet points, emojis, or code blocks."
        )
        self.tools = [open_website, open_application]
        self.available_tools = {
            "open_website": open_website,
            "open_application": open_application,
        }
        self.context_image_bytes: Optional[bytes] = None
        self.on_context_changed: Optional[Callable[[Optional[bytes]], None]] = None
        self.on_reply_generated: Optional[Callable[[str], None]] = None

    def set_context_image(self, image_bytes: bytes):
        """Stores the most recent captured screen context image bytes."""
        self.context_image_bytes = image_bytes
        if self.on_context_changed:
            try:
                self.on_context_changed(image_bytes)
            except Exception as e:
                print(f"[WARN] Error in context changed callback: {e}")

    def clear_context_image(self):
        """Clears currently stored context image."""
        self.context_image_bytes = None
        if self.on_context_changed:
            try:
                self.on_context_changed(None)
            except Exception as e:
                print(f"[WARN] Error in context changed callback: {e}")

    def query(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_reply_generated: Optional[Callable[[str], None]] = None,
    ):
        """Sends prompt (and optional image) to local Ollama in a background thread and speaks the response."""
        if not prompt or not prompt.strip():
            return

        effective_image = image_bytes if image_bytes is not None else self.context_image_bytes
        # One-time context consumption: clear staged context image immediately upon consumption
        if self.context_image_bytes is not None:
            self.clear_context_image()

        threading.Thread(
            target=self._process_query,
            args=(prompt.strip(), effective_image, on_status_change, on_reply_generated),
            daemon=True,
        ).start()

    def _process_query(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        on_status_change: Optional[Callable[[str], None]],
        on_reply_generated: Optional[Callable[[str], None]] = None,
    ):
        try:
            if on_status_change:
                on_status_change("THINKING")

            if image_bytes:
                img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                user_message: Dict[str, Any] = {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            else:
                user_message = {
                    "role": "user",
                    "content": prompt,
                }

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": self.system_instruction},
                user_message,
            ]

            target_model = self.vision_model if image_bytes else self.model

            # Execute chat with registered system tools
            try:
                response = self.client.chat(
                    model=target_model,
                    messages=messages,
                    tools=self.tools if not image_bytes else None,
                    options={
                        "temperature": 0.6,
                        "num_predict": 100,
                    },
                )
            except Exception as err:
                if "does not support tools" in str(err).lower():
                    response = self.client.chat(
                        model=target_model,
                        messages=messages,
                        options={
                            "temperature": 0.6,
                            "num_predict": 100,
                        },
                    )
                else:
                    raise err

            # Check if model requested tool execution
            tool_calls = getattr(response.message, "tool_calls", None)
            if not tool_calls and isinstance(response, dict) and "message" in response:
                tool_calls = response["message"].get("tool_calls", None)

            if tool_calls:
                for tool_call in tool_calls:
                    if hasattr(tool_call, "function"):
                        func_name = tool_call.function.name
                        func_args = tool_call.function.arguments or {}
                    elif isinstance(tool_call, dict) and "function" in tool_call:
                        func_name = tool_call["function"].get("name", "")
                        func_args = tool_call["function"].get("arguments", {})
                    else:
                        continue

                    tool_func = self.available_tools.get(func_name)
                    if tool_func:
                        try:
                            result_text = tool_func(**func_args)
                        except Exception as te:
                            result_text = f"Error executing {func_name}: {te}"

                        msg_payload = response.message if hasattr(response, "message") else response["message"]
                        messages.append(msg_payload)
                        messages.append({
                            "role": "tool",
                            "content": str(result_text),
                        })

                # Follow-up query to generate final natural spoken response
                final_response = self.client.chat(
                    model=target_model,
                    messages=messages,
                    options={"temperature": 0.3, "num_predict": 40},
                )
                if hasattr(final_response, "message") and hasattr(final_response.message, "content"):
                    reply = final_response.message.content.strip()
                elif isinstance(final_response, dict) and "message" in final_response:
                    reply = final_response["message"].get("content", "").strip()
                else:
                    reply = str(final_response).strip()
            else:
                if hasattr(response, "message") and hasattr(response.message, "content"):
                    reply = response.message.content.strip()
                elif isinstance(response, dict) and "message" in response and "content" in response["message"]:
                    reply = response["message"]["content"].strip()
                else:
                    reply = str(response).strip()

            if not reply:
                reply = "I didn't receive a response."

            if on_reply_generated:
                try:
                    on_reply_generated(reply)
                except Exception as cb_err:
                    print(f"[WARN] Error in on_reply_generated: {cb_err}")
            elif self.on_reply_generated:
                try:
                    self.on_reply_generated(reply)
                except Exception as cb_err:
                    print(f"[WARN] Error in self.on_reply_generated: {cb_err}")

            if on_status_change:
                on_status_change("SPEAKING")

            self._speak(reply)

            if on_status_change:
                on_status_change("IDLE")

        except Exception as e:
            print(f"[ERROR] Ollama request failed: {e}")
            error_msg = "Sorry, I had trouble connecting to the local model."
            if on_reply_generated:
                try:
                    on_reply_generated(error_msg)
                except Exception:
                    pass
            elif self.on_reply_generated:
                try:
                    self.on_reply_generated(error_msg)
                except Exception:
                    pass

            if on_status_change:
                on_status_change("SPEAKING")
            self._speak(error_msg)
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
GeminiAssistant = AIAssistant

