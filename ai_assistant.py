"""
Dual-Tier AI Assistant Engine for FRIDAY with native macOS Computer Use GUI Automation.
Primary: OpenAI API (gpt-4o for GUI perception-action agent loop, gpt-4o-mini for conversation & tools).
Fallback: Local Ollama (qwen2.5vl:3b for offline and network resilience).
GUI Automation: Native macOS Quartz CoreGraphics, AppleScript, and in-memory screenshots.
Local Speech: Native macOS say Text-to-Speech synthesis.
"""

import os
import json
import base64
import subprocess
import threading
import time
from typing import Optional, Callable, Dict, Any, List

from PyQt6.QtCore import QObject, pyqtSignal
from dotenv import load_dotenv
import openai
from openai import OpenAI
import ollama

from system_tools import open_website, open_application
from computer_controller import MacComputerController
from memory_vault import MemoryVault

# Load environment variables (.env)
load_dotenv()


class AIAssistantSignals(QObject):
    annotations_ready = pyqtSignal(list, str)  # (annotations, spoken_response)

# Complete tool definitions for OpenAI function calling
COMPUTER_USE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "computer_click",
            "description": "Click or double-click the mouse at normalized coordinates (x, y) on a 0-1000 grid",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Horizontal coordinate on a 0-1000 normalized grid (0 = left edge, 1000 = right edge)",
                    },
                    "y": {
                        "type": "integer",
                        "description": "Vertical coordinate on a 0-1000 normalized grid (0 = top edge, 1000 = bottom edge)",
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Mouse button to click (default: left)",
                    },
                    "double_click": {
                        "type": "boolean",
                        "description": "Whether to perform a double click (default: false)",
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_type",
            "description": "Type text into the currently focused window or field via native clipboard paste injection",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The exact string of text to type or paste"},
                    "press_enter": {
                        "type": "boolean",
                        "description": "Whether to press the Return/Enter key immediately after typing (default: false)",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_key",
            "description": "Press a special keyboard key or keyboard shortcut combination",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": ["enter", "escape", "tab", "space", "backspace", "up", "down", "left", "right"],
                        "description": "The key name to press",
                    },
                    "modifiers": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["command", "shift", "control", "option"]},
                        "description": "Modifier keys to hold down (e.g. ['command'] for Cmd+Key)",
                    },
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_scroll",
            "description": "Scroll the active window or view vertically",
            "parameters": {
                "type": "object",
                "properties": {
                    "delta_y": {
                        "type": "integer",
                        "description": "Scroll amount (positive = scroll up, negative = scroll down)",
                    }
                },
                "required": ["delta_y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_website",
            "description": "Open a website URL in the user's default web browser",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full web address URL (e.g. https://www.google.com)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Launch or activate a macOS application by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "The macOS application name (e.g. Safari, Notes, Terminal, Spotify)"},
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "annotate_screen",
            "strict": True,
            "description": "Draw colored visual bounding boxes and auto-positioned explanation cards across the user's screen. Use when the user asks to explain diagrams, circuits, code, or locate UI elements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spoken_response": {
                        "type": "string",
                        "description": "Brief 1-2 sentence spoken explanation for TTS audio.",
                    },
                    "annotations": {
                        "type": "array",
                        "description": "List of 1 to 5 distinct visual bounding boxes with explanations. Must contain at least 1 annotation.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "box_2d": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "[ymin, xmin, ymax, xmax] on a 0-1000 normalized grid tightly enclosing the target visual component.",
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Short heading (e.g. 'Inputs', 'Logic Gate', 'Attention Layer').",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "1 clear sentence explaining this specific component.",
                                },
                                "color": {
                                    "type": "string",
                                    "description": "Hex color code (e.g., '#3b82f6', '#22c55e', '#ef4444', '#f59e0b', '#a855f7').",
                                },
                            },
                            "required": ["box_2d", "label", "text", "color"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["spoken_response", "annotations"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_computer_automation",
            "description": "Trigger this when the user asks to write code, edit text, type, click, or perform multi-step actions on their screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The exact multi-step task goal to achieve on the user's desktop."
                    },
                    "initial_action_plan": {
                        "type": "string",
                        "description": "Short explanation of the steps you plan to take."
                    }
                },
                "required": ["goal"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_memory",
            "description": "Save important user facts, personal preferences, project details, or explicit notes into the local markdown memory vault for long-term recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category_file": {
                        "type": "string",
                        "enum": ["user_profile.md", "projects.md", "notes.md"],
                        "description": "The markdown file to store the memory in."
                    },
                    "fact_or_preference": {
                        "type": "string",
                        "description": "Concise, factual statement to remember (e.g. 'User prefers dark mode', 'User sister birthday is June 4')."
                    }
                },
                "required": ["category_file", "fact_or_preference"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Complete the computer automation task and return a natural 1-sentence confirmation summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "1 concise sentence summarizing what was accomplished on the computer"},
                },
                "required": ["summary"],
            },
        },
    },
]

# Alias for standard conversation tool subset
OPENAI_TOOLS = COMPUTER_USE_TOOLS


class AIAssistant:
    """
    Dual-Tier AI Desktop Assistant for FRIDAY:
    1. Primary Tier: OpenAI API (gpt-4o-mini for general chat/router, gpt-5.4 for vision/computer use).
    2. Fallback Tier: Local Ollama (strictly qwen2.5vl:3b) when offline or API is unavailable.
    """

    def __init__(
        self,
        primary_model: str = "gpt-4o-mini",
        computer_use_model: str = "gpt-5.4",
        fallback_host: str = "http://localhost:11434",
    ):
        self.primary_model = primary_model
        self.computer_use_model = computer_use_model
        self.agent_model = computer_use_model
        # Strictly use qwen2.5vl:3b for offline fallback
        self.fallback_model = "qwen2.5vl:3b"
        self.fallback_host = fallback_host

        # Computer GUI controller
        self.controller = MacComputerController()

        # Long-Term Markdown Memory Vault
        self.memory_vault = MemoryVault()

        # In-Session Rolling Conversation History
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history_turns: int = 12

        # Autonomous Computer Control & Abort state
        self.is_controlling_desktop: bool = False
        self.abort_event = threading.Event()

        # Primary OpenAI Client
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.strip():
            self.openai_client: Optional[OpenAI] = OpenAI(api_key=openai_key.strip())
        else:
            self.openai_client = None

        # Fallback Local Ollama Client
        self.ollama_client = ollama.Client(host=self.fallback_host)

        # Dedicated Qt GUI Signals
        self.signals = AIAssistantSignals()

        self.system_instruction = (
            "You are FRIDAY, an autonomous desktop assistant with full macOS GUI automation (Computer Use), Long-Term Memory, and AR visual annotation capabilities.\n\n"
            f"Screen dimensions: {self.controller.screen_width}x{self.controller.screen_height}.\n\n"
            "VISUAL COORDINATE REFERENCE:\n"
            "- The full-screen screenshot includes a subtle reference coordinate grid labeled from 0 to 1000 along both axes.\n"
            "- Vertical grid lines mark X coordinates: 100, 200, 300, ..., 900.\n"
            "- Horizontal grid lines mark Y coordinates: 100, 200, 300, ..., 900.\n"
            "- Inspect these labeled grid lines to accurately locate targets and emit precise [ymin, xmin, ymax, xmax] bounding boxes or click coordinates.\n\n"
            "VISUAL ANNOTATION GUIDELINES:\n"
            "1. BOUNDING BOX ACCURACY:\n"
            "   - Use the reference coordinate grid lines (0 to 1000) on the screenshot to pinpoint the exact boundaries of components.\n"
            "   - `box_2d` must be [ymin, xmin, ymax, xmax] tightly enclosing the specific visual element without capturing excess background.\n"
            "   - Avoid creating overlapping or nested bounding boxes unless one component is strictly a sub-element of another.\n"
            "   - Emit between 1 and 5 focused, distinct annotations. Even a single well-placed annotation is sufficient if there is only one relevant element on screen.\n"
            "2. CARD LABELS & TEXT:\n"
            "   - Keep `label` short (1-3 words).\n"
            "   - Keep `text` concise and direct (1 clear sentence).\n"
            "   - Use distinct contrasting colors for different functional groups (e.g. green '#22c55e' for inputs, blue '#3b82f6' for logic/processing, red '#ef4444' for outputs/critical blocks).\n\n"
            "DECISION GUIDELINES:\n"
            "1. DESKTOP INTERACTION VS. DIRECT ANSWER:\n"
            "   - If the user asks for code, writing, editing, typing, clicking, or actions relative to something on their screen (e.g., 'write this code below the hello world statement in my editor', 'clear the text in this compiler', 'click on cell B3', 'reply to this message', 'open app and do X'):\n"
            "     -> You MUST call `start_computer_automation` to inspect the screen and execute the typing/clicks directly into their app.\n"
            "   - If the user asks you to remember, save, or store a personal fact, preference, note, or project detail:\n"
            "     -> Call `save_user_memory` with appropriate category_file and fact_or_preference.\n"
            "   - If the user asks a purely theoretical question, general knowledge, translation, or conversational query with no screen interaction:\n"
            "     -> Answer directly in 1-2 concise sentences for spoken audio.\n"
            "   - If the user asks to visually explain a diagram, architecture, circuit, or find/highlight items on screen:\n"
            "     -> Call `annotate_screen`.\n\n"
            "2. COMPUTER USE AUTONOMY:\n"
            "   - Always verify each action in the following screenshot.\n"
            "   - Execute tasks with complete autonomy from start to finish. Once the goal is completed, call `finish_task`.\n\n"
            "3. Spoken Audio: Respond naturally and concisely in 1 to 2 sentences suitable for spoken audio. Never use markdown formatting, asterisks, bullet points, emojis, or code blocks."
        )

        self.available_tools = {
            "open_website": open_website,
            "open_application": open_application,
            "computer_click": self.controller.click,
            "computer_type": self.controller.type_text,
            "computer_key": self.controller.key_press,
            "computer_scroll": self.controller.scroll,
            "save_user_memory": lambda category_file, fact_or_preference: self.memory_vault.save_memory(category_file, fact_or_preference),
        }
        self.tools = [open_website, open_application]

        self.context_image_bytes: Optional[bytes] = None
        self.on_context_changed: Optional[Callable[[Optional[bytes]], None]] = None
        self.on_reply_generated: Optional[Callable[[str], None]] = None
        self.on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None

        # Concurrency & busy state locking
        self.is_busy: bool = False
        self.busy_lock = threading.Lock()

    def clear_history(self):
        """Clears in-session rolling conversation history."""
        self.conversation_history.clear()

    def abort_computer_agent(self):
        """Signals the computer agent loop to immediately halt operations."""
        print("[COMPUTER AGENT] Abort requested by user gesture!")
        self.abort_event.set()

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

    def _is_visual_query(self, prompt: str) -> bool:
        """Determines if a prompt implies inspecting or explaining visual elements on the desktop."""
        keywords = [
            "screen", "diagram", "circuit", "ui", "look at", "what is this", "what's this",
            "explain this", "where is", "find", "highlight", "annotate", "code", "window",
            "display", "image", "chart", "table", "button", "icon", "read this", "summarize this",
            "what am i looking at", "on my screen", "what do you see", "show me", "point to"
        ]
        p_lower = prompt.lower()
        return any(k in p_lower for k in keywords)

    def classify_query(self, user_query: str) -> Dict[str, Any]:
        """
        Fast pre-flight query router using gpt-4o-mini structured JSON (with local heuristic fallback):
        - target_model: 'gpt-4o-mini' or 'gpt-5.4'
        - requires_screen_context: bool (True only if query inspects on-screen diagrams, code, UI, or windows)
        - is_computer_use: bool (True if physical GUI action requested)
        - requires_memory_retrieval: bool (True if referencing user facts/preferences)
        - memory_search_query: str (Search terms for MemoryVault RAG)
        """
        if not user_query or not user_query.strip():
            return {
                "target_model": self.primary_model,
                "requires_screen_context": False,
                "is_computer_use": False,
                "requires_memory_retrieval": False,
                "memory_search_query": "",
            }

        # Attempt fast gpt-4o-mini structured triage call if online
        if self.openai_client is not None:
            try:
                router_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a fast, low-latency triage router for FRIDAY, an AI desktop assistant. "
                            "Analyze the user's utterance and return a JSON object with:\n"
                            "- \"target_model\": \"gpt-4o-mini\" (for conversation, facts, math, basic questions, memory recall, fast tool calling) or \"gpt-5.4\" (for deep visual inspection of diagrams/circuits, code on screen, complex UI annotation).\n"
                            "- \"requires_screen_context\": boolean. Set to TRUE ONLY if the user is asking about visual content currently visible on their screen (e.g. 'what is on my screen', 'explain this diagram', 'read the text in this window', 'where is the button'). Set to FALSE for math (e.g. 'what is 7+7'), general knowledge, conversations, memory queries, or questions with no visual reference.\n"
                            "- \"is_computer_use\": boolean. Set to TRUE if the user asks you to physically click, type, automate an application, search Google/YouTube, or control their desktop.\n"
                            "- \"requires_memory_retrieval\": boolean. Set to TRUE if the user asks about their personal info, preferences, past saved notes, or asks you to remember something.\n"
                            "- \"memory_search_query\": string. Concise keywords for memory lookup (or empty string if not needed)."
                        )
                    },
                    {"role": "user", "content": user_query}
                ]
                resp = self.openai_client.chat.completions.create(
                    model=self.primary_model,
                    messages=router_messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=80,
                    temperature=0.0,
                )
                result = json.loads(resp.choices[0].message.content or "{}")
                target_model = result.get("target_model", self.primary_model)
                if target_model not in ("gpt-4o-mini", "gpt-5.4"):
                    target_model = self.primary_model
                return {
                    "target_model": target_model,
                    "requires_screen_context": bool(result.get("requires_screen_context", False)),
                    "is_computer_use": bool(result.get("is_computer_use", False)),
                    "requires_memory_retrieval": bool(result.get("requires_memory_retrieval", False)),
                    "memory_search_query": str(result.get("memory_search_query", "")),
                }
            except Exception as e:
                print(f"[AI ROUTER WARN] gpt-4o-mini router call failed ({e}), using heuristic fallback.")

        # Local Heuristic Fallback
        is_computer = self._is_computer_control_task(user_query)
        is_visual = self._is_visual_query(user_query)
        is_memory = any(k in user_query.lower() for k in [
            "name", "birthday", "prefer", "favorite", "remember", "saved",
            "who am i", "who i am", "what do you know", "notes", "profile", "sister", "brother"
        ])
        return {
            "target_model": self.computer_use_model if is_visual else self.primary_model,
            "requires_screen_context": is_visual,
            "is_computer_use": is_computer,
            "requires_memory_retrieval": is_memory,
            "memory_search_query": user_query if is_memory else "",
        }

    def query(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_reply_generated: Optional[Callable[[str], None]] = None,
        on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ):
        """Sends prompt (with selective screen context) in a background thread and speaks the response."""
        if not prompt or not prompt.strip():
            return

        with self.busy_lock:
            if self.is_busy:
                print(f"[AI] Assistant is currently busy. Discarding overlapping query: '{prompt.strip()}'")
                return
            self.is_busy = True

        effective_image = image_bytes if image_bytes is not None else self.context_image_bytes

        # One-time context consumption: clear staged context image immediately upon staging
        if self.context_image_bytes is not None:
            self.clear_context_image()

        threading.Thread(
            target=self._process_query,
            args=(prompt.strip(), effective_image, on_status_change, on_reply_generated, on_annotations_generated),
            daemon=True,
        ).start()

    def _is_computer_control_task(self, prompt: str) -> bool:
        """Determines if a prompt requires autonomous perception-action computer use."""
        keywords = [
            "click", "double click", "press", "type in", "type into", "write in",
            "search on google", "search google for", "search youtube for", "play on spotify",
            "pause music", "control desktop", "scroll down", "scroll up", "fill out",
            "navigate to", "open notes and", "close window", "take a note", "go to",
            "compose", "in the search bar", "search bar", "in the subject", "in the content",
            "first link", "second link", "enter", "hit enter"
        ]
        p_lower = prompt.lower()
        return any(k in p_lower for k in keywords)

    def _process_query(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        on_status_change: Optional[Callable[[str], None]],
        on_reply_generated: Optional[Callable[[str], None]] = None,
        on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ):
        try:
            # 1. Pre-Flight Intent Classification
            classification = self.classify_query(prompt)
            print(f"[AI ROUTER] Triage: target={classification.get('target_model')}, screen={classification.get('requires_screen_context')}, computer_use={classification.get('is_computer_use')}, memory={classification.get('requires_memory_retrieval')}")

            # 2. Check for Computer Use GUI Automation
            if classification.get("is_computer_use", False) or self._is_computer_control_task(prompt):
                if self.openai_client is not None:
                    summary = self._execute_computer_agent_loop(prompt, on_status_change, on_reply_generated)
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": summary or "Completed desktop task."})
                    if len(self.conversation_history) > self.max_history_turns * 2:
                        self.conversation_history = self.conversation_history[-self.max_history_turns * 2:]
                    return
                else:
                    fallback_msg = "Computer automation requires an active cloud connection and is unavailable offline."
                    print(f"[AI] {fallback_msg}")
                    if on_reply_generated:
                        on_reply_generated(fallback_msg)
                    elif self.on_reply_generated:
                        self.on_reply_generated(fallback_msg)
                    if on_status_change:
                        on_status_change("SPEAKING")
                    self._speak(fallback_msg)
                    return

            # 3. Selective Screen Capture: Only capture screenshot if needed and not already supplied
            effective_image = image_bytes
            if effective_image is None and classification.get("requires_screen_context", False):
                try:
                    b64_snap, _, _ = self.controller.capture_screen_base64()
                    effective_image = base64.b64decode(b64_snap)
                    print(f"[AI] Selective Screen Capture: Acquired live full-screen context for '{prompt}'")
                except Exception as e:
                    print(f"[AI WARN] Selective screen capture error: {e}")

            if on_status_change:
                on_status_change("THINKING")

            reply: Optional[str] = None
            target_model = classification.get("target_model", self.primary_model)
            if effective_image is not None:
                target_model = self.computer_use_model

            # 4. Attempt Primary OpenAI API Dispatch
            if self.openai_client is not None:
                try:
                    reply = self._query_openai(
                        prompt,
                        effective_image,
                        on_annotations_generated,
                        on_status_change=on_status_change,
                        on_reply_generated=on_reply_generated,
                        target_model=target_model,
                        classification=classification,
                    )
                except Exception as e:
                    print(f"[AI] OpenAI unavailable ({e}), falling back to local Qwen 3B...")
                    reply = None

            # 5. Fallback to Local Ollama Engine (Strict Qwen 3B)
            if reply is None:
                reply = self._query_ollama(prompt, effective_image, classification=classification)

            if not reply or not reply.strip():
                reply = "I didn't receive a response."

            clean_reply = reply.strip()

            # Record in rolling in-session conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": clean_reply})
            if len(self.conversation_history) > self.max_history_turns * 2:
                self.conversation_history = self.conversation_history[-self.max_history_turns * 2:]

            if on_reply_generated:
                try:
                    on_reply_generated(clean_reply)
                except Exception as cb_err:
                    print(f"[WARN] Error in on_reply_generated: {cb_err}")
            elif self.on_reply_generated:
                try:
                    self.on_reply_generated(clean_reply)
                except Exception as cb_err:
                    print(f"[WARN] Error in self.on_reply_generated: {cb_err}")

            if on_status_change:
                on_status_change("SPEAKING")

            self._speak(clean_reply)

        except Exception as e:
            print(f"[ERROR] AI Assistant request failed: {e}")
            error_msg = "Sorry, I had trouble processing that request."
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
        finally:
            with self.busy_lock:
                self.is_busy = False
            if on_status_change:
                on_status_change("IDLE")

    def run_computer_agent(
        self,
        task_prompt: str,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_reply_generated: Optional[Callable[[str], None]] = None,
        max_iterations: int = 10,
    ) -> str:
        """
        Public entry point for autonomous Perception-Action loop.
        Applies concurrency locks if invoked directly outside query().
        """
        with self.busy_lock:
            if self.is_busy:
                print(f"[AI] Assistant is currently busy. Discarding computer task: '{task_prompt}'")
                return "Assistant is busy."
            self.is_busy = True

        try:
            return self._execute_computer_agent_loop(
                task_prompt, on_status_change, on_reply_generated, max_iterations
            )
        finally:
            with self.busy_lock:
                self.is_busy = False
            if on_status_change:
                on_status_change("IDLE")

    def _execute_computer_agent_loop(
        self,
        task_prompt: str,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_reply_generated: Optional[Callable[[str], None]] = None,
        max_iterations: int = 10,
    ) -> str:
        """
        Autonomous Perception-Action loop using gpt-4o vision to control macOS GUI.
        Perceives screen -> Predicts action -> Executes -> Verifies screenshot outcome -> Repeats until finish_task().
        """
        if not self.openai_client:
            fallback_msg = "Computer automation requires an active cloud connection and is unavailable offline."
            if on_reply_generated:
                on_reply_generated(fallback_msg)
            if on_status_change:
                on_status_change("SPEAKING")
            self._speak(fallback_msg)
            return fallback_msg

        self.is_controlling_desktop = True
        self.abort_event.clear()

        if on_status_change:
            on_status_change("CONTROLLING")

        print(f"\n[COMPUTER AGENT] Starting desktop GUI automation task: '{task_prompt}'")
        print(f"[COMPUTER AGENT] Display logical resolution: {self.controller.logical_width}x{self.controller.logical_height}")

        summary = "Completed desktop task."

        try:
            # Capture initial frame to anchor initial perception
            b64_init, init_w, init_h = self.controller.capture_screen_base64()

            system_prompt = (
                f"You are an expert desktop GUI automation agent interacting with a macOS screen of logical resolution {self.controller.screen_width}x{self.controller.screen_height}.\n\n"
                "COORDINATE SYSTEM & VISUAL REFERENCE:\n"
                "- Provide all coordinates (x, y) on a normalized 0 to 1000 scale:\n"
                "  * (0, 0) = Top-Left corner\n"
                "  * (500, 500) = Exact center of the screen\n"
                "  * (1000, 1000) = Bottom-Right corner\n"
                "  * Typical browser search/address bars are located horizontally centered near the top: (500, 75) to (500, 120).\n"
                "- The full-screen screenshot includes a subtle reference coordinate grid labeled from 0 to 1000 along both axes:\n"
                "  * Vertical grid lines mark X coordinates: 100, 200, 300, ..., 900.\n"
                "  * Horizontal grid lines mark Y coordinates: 100, 200, 300, ..., 900.\n"
                "- Inspect these labeled grid lines to accurately locate targets and emit precise (x, y) click coordinates or [ymin, xmin, ymax, xmax] bounding boxes.\n\n"
                "COMPUTER USE AGENT INSTRUCTIONS:\n"
                "1. END-TO-END AUTONOMY: When given a multi-step request (e.g. 'Open Chrome, go to YouTube, and search jazz', or 'type in search bar, press enter, and click first link'):\n"
                "   - Execute all necessary actions sequentially across iterations.\n"
                "   - DO NOT stop after step 1 to ask for user confirmation or input.\n"
                "   - Continue iterating through the perception-action loop until the entire objective is completed.\n"
                "   - Only call `finish_task` when the final goal is fully achieved and visible on screen.\n\n"
                "2. MANDATORY VISUAL VERIFICATION:\n"
                "   - On every iteration (from iteration 2 onwards), first inspect the latest screenshot to verify that your previous action produced the intended UI change.\n"
                "   - If you clicked a search box, verify that the cursor is active before typing.\n"
                "   - If you typed text, confirm the text actually appears in the field before pressing Enter.\n"
                "   - If an action missed or failed, adjust your normalized coordinates and retry immediately.\n\n"
                "3. Take discrete actions per turn (click, type, or press key) to allow observation on the next iteration.\n"
                "4. When the user's goal is fully achieved, call finish_task(summary='...') with a concise 1-sentence confirmation."
            )

            init_img_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_init}"
                }
            }

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Task: {task_prompt}\nInitial screen state:"},
                        init_img_content,
                    ]
                }
            ]

            for iteration in range(max_iterations):
                if self.abort_event.is_set():
                    print("[COMPUTER AGENT] Abort signal received. Halting computer agent loop immediately.")
                    summary = "Desktop control aborted."
                    break

                print(f"\n[COMPUTER AGENT] --- Iteration {iteration + 1}/{max_iterations} ---")
                print(f"[COMPUTER AGENT] Querying {self.computer_use_model} for next UI action...")

                try:
                    try:
                        response = self.openai_client.chat.completions.create(
                            model=self.computer_use_model,
                            messages=messages,
                            tools=COMPUTER_USE_TOOLS,
                            tool_choice="auto",
                            max_completion_tokens=350,
                            temperature=0.2,
                        )
                    except Exception as param_err:
                        if "temperature" in str(param_err).lower() or "unsupported_parameter" in str(param_err).lower():
                            response = self.openai_client.chat.completions.create(
                                model=self.computer_use_model,
                                messages=messages,
                                tools=COMPUTER_USE_TOOLS,
                                tool_choice="auto",
                                max_completion_tokens=350,
                            )
                        else:
                            raise param_err
                except Exception as e:
                    print(f"[COMPUTER AGENT ERROR] Model inference failed at step {iteration+1}: {e}")
                    break

                if self.abort_event.is_set():
                    print("[COMPUTER AGENT] Abort signal received after inference. Halting loop.")
                    summary = "Desktop control aborted."
                    break

                msg = response.choices[0].message
                messages.append(msg)

                if not msg.tool_calls:
                    text_content = msg.content or ""
                    print(f"[COMPUTER AGENT] Model returned text without tool calls: {text_content}")
                    if iteration < max_iterations - 1:
                        b64_next, next_w, next_h = self.controller.capture_screen_base64()
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Continue executing the next action to achieve the goal. Do not stop until calling finish_task.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64_next}"
                                    }
                                }
                            ]
                        })
                        continue
                    else:
                        summary = text_content or "Completed desktop task."
                        break

                finished = False
                print(f"[COMPUTER AGENT] Model emitted {len(msg.tool_calls)} action(s): {[tc.function.name for tc in msg.tool_calls]}")

                for tool_call in msg.tool_calls:
                    if self.abort_event.is_set():
                        print("[COMPUTER AGENT] Abort signal received before tool execution. Halting loop.")
                        summary = "Desktop control aborted."
                        finished = True
                        break

                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except Exception:
                        args = {}

                    print(f"[COMPUTER AGENT] Executing action: {func_name}({args})")

                    if func_name == "finish_task":
                        summary = args.get("summary", "Task completed.")
                        finished = True
                        tool_result = f"Task marked finished: {summary}"
                    elif func_name in self.available_tools:
                        func = self.available_tools[func_name]
                        try:
                            tool_result = func(**args)
                        except Exception as te:
                            tool_result = f"Error executing {func_name}: {te}"
                    else:
                        tool_result = f"Unknown tool: {func_name}"

                    print(f"[COMPUTER AGENT] Action result: {tool_result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"{tool_result}. Observe the next screenshot to verify outcome.",
                    })

                # Add settling delay so UI animations, focus changes, and popups render before capture
                time.sleep(0.35)

                if finished or self.abort_event.is_set():
                    if self.abort_event.is_set():
                        summary = "Desktop control aborted."
                    print(f"[COMPUTER AGENT] Goal achieved / loop ended on iteration {iteration + 1}!")
                    break

                # Capture updated live screenshot for next iteration if loop continues
                if iteration < max_iterations - 1:
                    b64_next, next_w, next_h = self.controller.capture_screen_base64()
                    print(f"[COMPUTER AGENT] Live screenshot updated: {next_w}x{next_h} px")
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"[Step {iteration+2}]: Updated live screenshot. Verify outcome of previous action and determine next step.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_next}"
                                }
                            }
                        ]
                    })
        finally:
            self.is_controlling_desktop = False

        print(f"[COMPUTER AGENT] Final Summary: {summary}\n")

        # Presentation and speech
        if on_reply_generated:
            try:
                on_reply_generated(summary)
            except Exception:
                pass
        elif self.on_reply_generated:
            try:
                self.on_reply_generated(summary)
            except Exception:
                pass

        if on_status_change:
            on_status_change("SPEAKING")

        self._speak(summary)

        return summary

    def _query_openai(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_reply_generated: Optional[Callable[[str], None]] = None,
        target_model: Optional[str] = None,
        classification: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Dispatches query to OpenAI gpt-4o-mini / gpt-5.4 with selective RAG memory and multimodal vision."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        # Selective BM25/keyword memory retrieval
        req_mem = classification.get("requires_memory_retrieval", True) if classification else True
        search_q = classification.get("memory_search_query", "") if classification else ""
        if not search_q:
            search_q = prompt

        vault_context = self.memory_vault.retrieve_relevant_memories(search_q, top_k=3) if req_mem else ""

        if vault_context:
            system_content = f"{self.system_instruction}\n\n---\nRELEVANT LONG-TERM MEMORIES:\n{vault_context}\n---"
        else:
            system_content = self.system_instruction

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        # Append recent rolling in-session conversation history
        for turn in self.conversation_history:
            messages.append({"role": turn["role"], "content": str(turn["content"])})

        if image_bytes:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            user_content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    },
                },
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": prompt})

        chosen_model = target_model or (self.computer_use_model if image_bytes else self.primary_model)
        print(f"[AI] Dispatching query to {chosen_model} (multimodal={image_bytes is not None}, memories={bool(vault_context)})...")

        try:
            response = self.openai_client.chat.completions.create(
                model=chosen_model,
                messages=messages,
                tools=COMPUTER_USE_TOOLS,
                tool_choice="auto",
                max_completion_tokens=400,
                temperature=0.2,
            )
        except Exception as err:
            if "temperature" in str(err).lower() or "unsupported_parameter" in str(err).lower():
                response = self.openai_client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    tools=COMPUTER_USE_TOOLS,
                    tool_choice="auto",
                    max_completion_tokens=400,
                )
            else:
                raise err

        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            spoken_summary = None

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except Exception:
                    func_args = {}

                if func_name == "start_computer_automation":
                    goal = func_args.get("goal", prompt)
                    print(f"[AI] Model initiated start_computer_automation: goal='{goal}'")
                    return self._execute_computer_agent_loop(
                        task_prompt=goal,
                        on_status_change=on_status_change,
                        on_reply_generated=on_reply_generated,
                    )
                elif func_name == "annotate_screen":
                    spoken_summary = func_args.get("spoken_response", "Here is the visual breakdown.")
                    anns = func_args.get("annotations", [])
                    print(f"[AI] annotate_screen tool triggered with {len(anns)} annotations")
                    if len(anns) == 0:
                        print("[AI WARNING] Received 0 annotations, falling back to full-card response.")
                    else:
                        try:
                            self.signals.annotations_ready.emit(anns, spoken_summary)
                        except Exception as sig_err:
                            print(f"[AI WARN] Failed to emit annotations_ready signal: {sig_err}")
                        if on_annotations_generated:
                            on_annotations_generated(anns)
                        elif self.on_annotations_generated:
                            on_annotations_generated(anns)
                    result_text = f"Annotations displayed on screen: {spoken_summary}"
                else:
                    tool_func = self.available_tools.get(func_name)
                    if tool_func:
                        try:
                            result_text = tool_func(**func_args)
                        except Exception as te:
                            result_text = f"Error executing {func_name}: {te}"
                    else:
                        result_text = f"Unknown function {func_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result_text),
                })

            if spoken_summary:
                return spoken_summary

            # Follow-up completion for natural spoken confirmation
            try:
                follow_up = self.openai_client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    max_completion_tokens=100,
                    temperature=0.2,
                )
                return follow_up.choices[0].message.content or ""
            except Exception:
                return "Completed request."
        else:
            return msg.content or ""

    def _query_ollama(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        classification: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Dispatches query strictly to local Ollama qwen2.5vl:3b as offline fallback."""
        # Fallback guardrail: check if computer automation was requested
        if self._is_computer_control_task(prompt):
            return "Computer automation requires an active cloud connection and is unavailable offline."

        req_mem = classification.get("requires_memory_retrieval", True) if classification else True
        search_q = classification.get("memory_search_query", "") if classification else ""
        if not search_q:
            search_q = prompt

        vault_context = self.memory_vault.retrieve_relevant_memories(search_q, top_k=3) if req_mem else ""

        if vault_context:
            system_content = f"{self.system_instruction}\n\n---\nRELEVANT LONG-TERM MEMORIES:\n{vault_context}\n---"
        else:
            system_content = self.system_instruction

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        # Append recent rolling in-session conversation history
        for turn in self.conversation_history:
            messages.append({"role": turn["role"], "content": str(turn["content"])})

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

        messages.append(user_message)

        try:
            response = self.ollama_client.chat(
                model=self.fallback_model,
                messages=messages,
                tools=self.tools if not image_bytes else None,
                options={"temperature": 0.6, "num_predict": 100},
            )
        except Exception as err:
            if "does not support tools" in str(err).lower():
                response = self.ollama_client.chat(
                    model=self.fallback_model,
                    messages=messages,
                    options={"temperature": 0.6, "num_predict": 100},
                )
            else:
                raise err

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

            final_response = self.ollama_client.chat(
                model=self.fallback_model,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 40},
            )
            if hasattr(final_response, "message") and hasattr(final_response.message, "content"):
                return final_response.message.content.strip()
            elif isinstance(final_response, dict) and "message" in final_response:
                return final_response["message"].get("content", "").strip()
            else:
                return str(final_response).strip()
        else:
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content.strip()
            elif isinstance(response, dict) and "message" in response and "content" in response["message"]:
                return response["message"]["content"].strip()
            else:
                return str(response).strip()

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

