"""
Dual-Tier AI Assistant Engine for Open FRIDAY.
Primary: OpenAI API (gpt-4o-mini for fast routing/chat, gpt-5.4 for visual perception, annotations & computer control).
Fallback: Local Ollama (qwen2.5vl:3b for offline resilience).
Features:
- State-of-the-art Prompt Engineering with Voice-First Output Contract and Unabridged Code Execution.
- Hybrid Computer Use & Sandboxed Bash Execution (`execute_bash`).
- Unambiguous Intent Routing (Visual AR Annotation vs. Desktop Computer Use).
- Set-of-Marks (SoM) Accessibility Grounding for 100% click accuracy.
- Semantic Vector RAG via FastEmbed MemoryVault.
- Step-by-Step Action Verification and Consecutive Action Stall Detection (3x repeat prevention).
- Robust Tool Argument Parsing & Large Token Budgets (4096 tokens) for smooth, complete code writing.
- Local speech synthesis with lifecycle tracking.
"""

import os
import json
import re
import base64
import subprocess
import threading
import time
from typing import Optional, Callable, Dict, Any, List, Tuple

from PyQt6.QtCore import QObject, pyqtSignal
from dotenv import load_dotenv
import openai
from openai import OpenAI
import ollama

from system_tools import open_website, open_application
from computer_controller import MacComputerController
from memory_vault import MemoryVault
from bash_executor import BashExecutor

# Load environment variables (.env)
load_dotenv()


def _safe_parse_tool_arguments(raw_args: str) -> Dict[str, Any]:
    """
    Robust JSON parser for OpenAI tool call arguments.
    Gracefully handles multiline strings, unescaped newlines/quotes in code, and truncated token streams.
    """
    if not raw_args or not raw_args.strip():
        return {}

    # 1. Standard JSON Parse
    try:
        return json.loads(raw_args)
    except Exception:
        pass

    # 2. Fix unescaped raw newlines / carriage returns
    try:
        fixed = raw_args.replace("\n", "\\n").replace("\r", "\\r")
        return json.loads(fixed)
    except Exception:
        pass

    # 3. Regex-based resilient extraction
    result: Dict[str, Any] = {}

    # Extract observation_and_verification
    obs_match = re.search(r'"observation_and_verification"\s*:\s*"([^"]*)', raw_args)
    if obs_match:
        result["observation_and_verification"] = obs_match.group(1).replace("\\n", "\n").replace('\\"', '"')

    # Extract text / code / content
    text_match = re.search(r'"(?:text|code|content|value|string)"\s*:\s*"([\s\S]*)', raw_args)
    if text_match:
        val = text_match.group(1)
        if val.endswith('"}'):
            val = val[:-2]
        elif val.endswith('"'):
            val = val[:-1]
        val = re.split(r'",\s*"', val)[0]
        result["text"] = val.replace("\\n", "\n").replace('\\"', '"')

    # Extract numeric fields
    for field in ["element_id", "x", "y", "delta_y", "radius"]:
        num_match = re.search(rf'"{field}"\s*:\s*(\d+)', raw_args)
        if num_match:
            result[field] = int(num_match.group(1))

    # Extract string fields
    for field in ["combo", "url", "app_name", "summary", "button", "key", "goal", "command"]:
        str_match = re.search(rf'"{field}"\s*:\s*"([^"]+)"', raw_args)
        if str_match:
            result[field] = str_match.group(1)

    # Extract boolean fields
    for field in ["press_enter", "double_click"]:
        bool_match = re.search(rf'"{field}"\s*:\s*(true|false)', raw_args, re.IGNORECASE)
        if bool_match:
            result[field] = bool_match.group(1).lower() == "true"

    return result


class AIAssistantSignals(QObject):
    annotations_ready = pyqtSignal(list, str)  # (annotations, spoken_response)
    speaking_started = pyqtSignal()
    speaking_finished = pyqtSignal()


# Conversation-Level Tools (Dispatched during standard queries & routing)
CONVERSATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "annotate_screen",
            "strict": True,
            "description": "Highlight and explain visual components, diagrams, circuits, code lines, or UI features across the screen with glowing bounding boxes and floating cards. Use whenever the user asks to explain, point out, find, or highlight something on their screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spoken_response": {
                        "type": "string",
                        "description": "Brief 1-sentence spoken summary for voice output.",
                    },
                    "annotations": {
                        "type": "array",
                        "description": "List of 1 to 6 distinct visual bounding boxes with short explanations.",
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
                                    "description": "Short heading (e.g. 'Input Layer', 'Logic Gate', 'Search Bar', 'Bug Location').",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "1 clear sentence explaining this component.",
                                },
                                "color": {
                                    "type": "string",
                                    "description": "Hex color (e.g. '#3b82f6', '#22c55e', '#ef4444', '#f59e0b', '#a855f7').",
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
            "name": "execute_bash",
            "description": "Execute a non-destructive macOS terminal command or osascript (AppleScript) snippet. Use this for deterministic actions like launching apps, file lookups, clipboard operations, web curl requests, or media controls instead of slow visual clicking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to execute (e.g. 'open -a \"Spotify\"', 'osascript -e \"tell application \\\"Finder\\\" to open POSIX file \\\"/Users\\\"\"', 'ls -la ~/Downloads')."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_computer_automation",
            "description": "Trigger this ONLY when the user asks you to actively click, type, write code in their editor, automate software, search websites, or perform multi-step desktop tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The exact multi-step task goal to achieve on the desktop."
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
            "name": "save_user_memory",
            "description": "Save important user facts, personal preferences, project details, or explicit notes into the local semantic memory vault for long-term recall.",
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
                        "description": "Concise, factual statement to remember (e.g. 'User prefers dark mode', 'Sister birthday is June 4')."
                    }
                },
                "required": ["category_file", "fact_or_preference"]
            }
        }
    },
]

# Action Tools available strictly INSIDE the autonomous Computer Agent Loop
# Each action requires mandatory visual observation and verification
COMPUTER_ACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a non-destructive macOS terminal command or osascript (AppleScript) snippet deterministically (e.g. launch apps with 'open -a', volume/media controls via osascript, file lookups, curl) instead of slow visual clicking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing current visual state and why this terminal/AppleScript command is needed.",
                    },
                    "command": {
                        "type": "string",
                        "description": "The exact shell command or osascript to execute.",
                    },
                },
                "required": ["command", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Click or double-click an interactive UI element by its discrete Set-of-Marks integer ID badge (e.g. [1], [2], [3]) with 100% precision. ALWAYS PREFER this over computer_click when an ID tag is visible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing what visual changes occurred after previous action, whether it succeeded, and why this click is needed.",
                    },
                    "element_id": {
                        "type": "integer",
                        "description": "The exact Set-of-Marks numeric ID visible on the element badge in the screenshot (e.g. 1, 2, 5)",
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
                "required": ["element_id", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_click",
            "description": "Click or double-click the mouse at normalized coordinates (x, y) on a 0-1000 grid. Use as fallback for unlabeled canvas elements, video players, or custom widgets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing what visual changes occurred after previous action, whether it succeeded, and why this coordinate click is needed.",
                    },
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
                "required": ["x", "y", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_hover",
            "description": "Hover mouse cursor over normalized (x, y) coordinates to reveal tooltips or flyout menus",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing what visual changes occurred and why hovering here is needed.",
                    },
                    "x": {"type": "integer", "description": "Horizontal coordinate on 0-1000 normalized grid"},
                    "y": {"type": "integer", "description": "Vertical coordinate on 0-1000 normalized grid"},
                },
                "required": ["x", "y", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_drag",
            "description": "Drag the mouse smoothly from (start_x, start_y) to (end_x, end_y) on a 0-1000 normalized grid",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing visual state and reason for dragging.",
                    },
                    "start_x": {"type": "integer", "description": "Starting X (0-1000)"},
                    "start_y": {"type": "integer", "description": "Starting Y (0-1000)"},
                    "end_x": {"type": "integer", "description": "Ending X (0-1000)"},
                    "end_y": {"type": "integer", "description": "Ending Y (0-1000)"},
                },
                "required": ["start_x", "start_y", "end_x", "end_y", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_type",
            "description": "Type or paste full, unabridged code or text into the currently focused window, text field, or notebook cell via fast clipboard injection (preserves user's prior clipboard history).",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences verifying that the target text box/cell is focused and explaining what will be typed.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The exact full string, text, or multi-line code to type or paste.",
                    },
                    "press_enter": {
                        "type": "boolean",
                        "description": "Whether to press Return/Enter immediately after typing (default: false)",
                    },
                },
                "required": ["text", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_hotkey",
            "description": "Press a macOS keyboard shortcut combination (e.g. 'cmd+t', 'cmd+w', 'cmd+shift+p', 'cmd+k', 'ctrl+c', 'option+tab', 'cmd+space')",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing current visual state and why this keyboard shortcut is needed.",
                    },
                    "combo": {"type": "string", "description": "The hotkey string (e.g. 'cmd+t', 'cmd+shift+p', 'cmd+k')"},
                },
                "required": ["combo", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_key",
            "description": "Press a special keyboard key (e.g. 'enter', 'escape', 'tab', 'space', 'backspace', 'up', 'down', 'left', 'right')",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing visual state and why pressing this key is needed.",
                    },
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
                "required": ["key", "observation_and_verification"],
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
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing visual state and reason for scrolling.",
                    },
                    "delta_y": {
                        "type": "integer",
                        "description": "Scroll amount (positive = scroll up, negative = scroll down)",
                    }
                },
                "required": ["delta_y", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_region",
            "description": "Foveated High-Resolution Perception: inspect an uncompressed 1:1 pixel zoomed crop around (x, y) to read small text, code lines, or tiny buttons",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences explaining what region needs high-resolution visual inspection.",
                    },
                    "x": {"type": "integer", "description": "Center X on 0-1000 normalized grid"},
                    "y": {"type": "integer", "description": "Center Y on 0-1000 normalized grid"},
                    "radius": {"type": "integer", "description": "Crop radius in logical points (default: 150)"},
                },
                "required": ["x", "y", "observation_and_verification"],
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
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing reason for opening URL.",
                    },
                    "url": {"type": "string", "description": "The full web address URL (e.g. https://www.google.com)"},
                },
                "required": ["url", "observation_and_verification"],
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
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences describing reason for launching app.",
                    },
                    "app_name": {"type": "string", "description": "The macOS application name (e.g. Safari, Notes, Terminal, Spotify)"},
                },
                "required": ["app_name", "observation_and_verification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Complete the computer automation task and return a natural 1-sentence confirmation summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "observation_and_verification": {
                        "type": "string",
                        "description": "MANDATORY: 1-2 sentences verifying that the goal was fully achieved based on the final screenshot.",
                    },
                    "summary": {"type": "string", "description": "1 concise sentence summarizing what was accomplished on the computer"},
                },
                "required": ["summary", "observation_and_verification"],
            },
        },
    },
]

# Compatibility aliases
COMPUTER_USE_TOOLS = CONVERSATION_TOOLS
OPENAI_TOOLS = CONVERSATION_TOOLS


class AIAssistant:
    """
    Dual-Tier AI Desktop Assistant for Open FRIDAY:
    1. Primary Tier: OpenAI API (gpt-4o-mini for general chat/router, gpt-5.4/gpt-4o for vision/computer use).
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
        self.fallback_model = "qwen2.5vl:3b"
        self.fallback_host = fallback_host

        # Computer GUI controller with Set-of-Marks grounding
        self.controller = MacComputerController()

        # Semantic Vector RAG Memory Vault
        self.memory_vault = MemoryVault()

        # Safe Non-Blocking Bash Executor
        self.bash_executor = BashExecutor(default_timeout=10.0)

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

        # State-of-the-Art Voice-First Prompt Engineering Contract
        self.system_instruction = (
            "You are Open FRIDAY, an ultra-fast, voice-first autonomous AI desktop assistant on macOS. When speaking conversationally with the user, refer to yourself simply as Friday.\n\n"
            f"Active Display Logical Resolution: {self.controller.screen_width}x{self.controller.screen_height}.\n\n"
            "=== CONVERSATIONAL VOICE OUTPUT CONTRACT ===\n"
            "- VOICE-FIRST CONCISENESS: Every direct text answer you emit is read aloud to the user via local speech synthesis (TTS). Keep all conversational answers natural, punchy, and strictly 1 to 2 sentences maximum.\n"
            "- NO CODE OR MARKDOWN IN SPEECH: Never recite raw code blocks, syntax, curly braces, imports, markdown tables, asterisks (**bold**), bullet points, headers, or URLs in conversational voice output.\n"
            "- CODE CREATION REQUESTS: If the user asks you to write code, create a script, or build something on their desktop, DO NOT speak the code aloud. Instead, immediately trigger `execute_bash` (to write/create files directly) or `start_computer_automation` (to type/paste it into their focused editor/notebook). If the user asks a theoretical programming question, explain the concept conceptually in 1-2 plain spoken English sentences.\n\n"
            "=== HYBRID AUTOMATION & DETERMINISTIC TOOL USAGE ===\n"
            "1. TERMINAL & APPLESCRIPT FIRST (`execute_bash`):\n"
            "   - Always prefer `execute_bash` for deterministic actions that can be executed via CLI or AppleScript (e.g. `open -a \"AppName\"`, writing files, file search `find`/`ls`, volume/media controls via `osascript`, `curl` requests, reading clipboard).\n"
            "2. DESKTOP GUI AUTOMATION (`start_computer_automation`):\n"
            "   - Call `start_computer_automation` when the user asks you to interact with graphical desktop interfaces, click buttons, fill out web forms, or control software lacking a CLI interface.\n"
            "3. ON-SCREEN VISUAL EXPLANATION (`annotate_screen`):\n"
            "   - When the user asks to explain, describe, point out, find, highlight, or break down diagrams, circuits, errors, or visual features visible on their screen:\n"
            "     -> You MUST call `annotate_screen` with normalized [ymin, xmin, ymax, xmax] bounding boxes (0-1000 scale), labels, concise text, and colors.\n"
            "     -> Provide a 1-sentence spoken summary in `spoken_response`.\n"
            "     -> NEVER call computer click/type tools for visual inspection requests.\n"
            "4. PERSONAL MEMORY VAULT (`save_user_memory`):\n"
            "   - When the user shares personal facts, preferences, project details, or says 'remember that...':\n"
            "     -> Call `save_user_memory` to store it in semantic RAG.\n"
            "5. SECURITY RESTRICTION:\n"
            "   - Root privilege elevation (`sudo`, `doas`) and destructive system commands are strictly prohibited."
        )

        self.available_tools = {
            "click_element": self.controller.click_element,
            "computer_click": self.controller.click,
            "computer_hover": self.controller.hover,
            "computer_drag": self.controller.drag,
            "computer_type": self.controller.type_text,
            "computer_key": self.controller.key_press,
            "computer_hotkey": self.controller.hotkey,
            "computer_scroll": self.controller.scroll,
            "inspect_region": self._handle_inspect_region,
            "execute_bash": self.bash_executor.run,
            "open_website": open_website,
            "open_application": open_application,
            "save_user_memory": lambda category_file, fact_or_preference: self.memory_vault.save_memory(category_file, fact_or_preference),
        }
        self.tools = [open_website, open_application, self.bash_executor.run]

        self.context_image_bytes: Optional[bytes] = None
        self.on_context_changed: Optional[Callable[[Optional[bytes]], None]] = None
        self.on_reply_generated: Optional[Callable[[str], None]] = None
        self.on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None

        # Concurrency & busy state locking
        self.is_busy: bool = False
        self.busy_lock = threading.Lock()

    def _handle_inspect_region(self, x: int, y: int, radius: int = 150) -> str:
        """Handles foveated high-resolution inspection requests."""
        b64_crop, cw, ch = self.controller.inspect_region(x, y, radius)
        return f"Acquired 1:1 high-resolution zoomed crop ({cw}x{ch} px) centered at norm ({x}, {y})."

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

    def _is_visual_annotation_query(self, prompt: str) -> bool:
        """Determines if a prompt is asking to explain, point out, annotate, or inspect screen content."""
        keywords = [
            "screen", "diagram", "circuit", "ui", "look at", "what is this", "what's this",
            "explain this", "where is", "find", "highlight", "annotate", "code on screen",
            "what am i looking at", "on my screen", "what do you see", "show me", "point to",
            "break down", "error on screen", "bug in this", "what does this mean", "describe this"
        ]
        p_lower = prompt.lower()
        return any(k in p_lower for k in keywords)

    def _is_computer_control_task(self, prompt: str) -> bool:
        """Determines if a prompt requires autonomous desktop automation (clicking, typing, opening)."""
        if self._is_visual_annotation_query(prompt) and not any(a in prompt.lower() for a in ["type in", "write in", "click on", "fill out", "write code", "paste code"]):
            return False

        keywords = [
            "click", "double click", "press", "type in", "type into", "write in", "write the code", "write code",
            "search on google", "search google for", "search youtube for", "play on spotify",
            "pause music", "control desktop", "scroll down", "scroll up", "fill out",
            "navigate to", "open notes and", "close window", "take a note", "go to",
            "compose", "in the search bar", "search bar", "in the subject", "in the content",
            "first link", "second link", "hit enter", "drag", "hover", "paste", "insert into"
        ]
        p_lower = prompt.lower()
        return any(k in p_lower for k in keywords)

    def classify_query(self, user_query: str) -> Dict[str, Any]:
        """Fast pre-flight query router with unambiguous visual annotation vs. computer control separation."""
        if not user_query or not user_query.strip():
            return {
                "target_model": self.primary_model,
                "requires_screen_context": False,
                "is_computer_use": False,
                "requires_memory_retrieval": False,
                "memory_search_query": "",
            }

        if self.openai_client is not None:
            try:
                router_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a fast, low-latency triage router for Open FRIDAY, an AI desktop assistant. "
                            "Analyze the user's utterance and return a JSON object with:\n"
                            "- \"target_model\": \"gpt-4o-mini\" or \"gpt-5.4\"\n"
                            "- \"requires_screen_context\": boolean. Set to TRUE whenever the user asks to explain, point out, highlight, locate, inspect, or understand visual content on their screen.\n"
                            "- \"is_computer_use\": boolean. Set to TRUE ONLY if the user asks you to actively click, type, write code on their desktop, automate software, open new tabs/apps, or control their computer.\n"
                            "- \"requires_memory_retrieval\": boolean. Set to TRUE if the user asks about past context, preferences, or notes.\n"
                            "- \"memory_search_query\": string. Search keywords for MemoryVault."
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

                req_screen = bool(result.get("requires_screen_context", False))
                is_comp = bool(result.get("is_computer_use", False))

                # Hard guardrail: If screen explanation/annotation was requested, disable computer use
                if req_screen and not any(k in user_query.lower() for k in ["click", "type", "write", "open app", "automate", "paste", "insert"]):
                    is_comp = False

                return {
                    "target_model": target_model,
                    "requires_screen_context": req_screen,
                    "is_computer_use": is_comp,
                    "requires_memory_retrieval": bool(result.get("requires_memory_retrieval", False)),
                    "memory_search_query": str(result.get("memory_search_query", "")),
                }
            except Exception as e:
                print(f"[AI ROUTER WARN] gpt-4o-mini router call failed ({e}), using heuristic fallback.")

        is_visual = self._is_visual_annotation_query(user_query)
        is_computer = self._is_computer_control_task(user_query)
        if is_visual and not any(k in user_query.lower() for k in ["write", "type", "click"]):
            is_computer = False

        is_memory = any(k in user_query.lower() for k in [
            "name", "birthday", "prefer", "favorite", "remember", "saved",
            "who am i", "who i am", "what do you know", "notes", "profile", "sister", "brother"
        ])
        return {
            "target_model": self.computer_use_model if (is_visual or is_computer) else self.primary_model,
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
        """Sends prompt in background thread and speaks response."""
        if not prompt or not prompt.strip():
            return

        with self.busy_lock:
            if self.is_busy:
                print(f"[AI] Assistant is currently busy. Discarding overlapping query: '{prompt.strip()}'")
                return
            self.is_busy = True

        effective_image = image_bytes if image_bytes is not None else self.context_image_bytes

        if self.context_image_bytes is not None:
            self.clear_context_image()

        threading.Thread(
            target=self._process_query,
            args=(prompt.strip(), effective_image, on_status_change, on_reply_generated, on_annotations_generated),
            daemon=True,
        ).start()

    def _process_query(
        self,
        prompt: str,
        image_bytes: Optional[bytes],
        on_status_change: Optional[Callable[[str], None]],
        on_reply_generated: Optional[Callable[[str], None]] = None,
        on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ):
        try:
            classification = self.classify_query(prompt)
            print(f"[AI ROUTER] Triage: target={classification.get('target_model')}, screen={classification.get('requires_screen_context')}, computer_use={classification.get('is_computer_use')}, memory={classification.get('requires_memory_retrieval')}")

            # 1. Computer Use Desktop Automation Loop
            if classification.get("is_computer_use", False) or self._is_computer_control_task(prompt):
                if self.openai_client is not None:
                    summary = self._execute_computer_agent_loop(prompt, on_status_change, on_reply_generated, max_iterations=30)
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

            # 2. Visual Inspection / AR Annotations (Screen capture if needed)
            effective_image = image_bytes
            if effective_image is None and classification.get("requires_screen_context", False):
                try:
                    b64_snap, _, _, _ = self.controller.capture_screen_base64(apply_grid=True, apply_som=False)
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

            if reply is None:
                reply = self._query_ollama(prompt, effective_image, classification=classification)

            if not reply or not reply.strip():
                reply = "I didn't receive a response."

            clean_reply = reply.strip()

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
        max_iterations: int = 30,
    ) -> str:
        """Public entry point for autonomous Set-of-Marks Perception-Action loop."""
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
        max_iterations: int = 30,
    ) -> str:
        """
        Autonomous Set-of-Marks (SoM) Perception-Action loop using COMPUTER_ACTION_TOOLS strictly,
        with step-by-step verification, resilient JSON parsing, 4096-token budget, and 3x consecutive identical action stall detection.
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

        print(f"\n[COMPUTER AGENT] Starting desktop GUI automation task (max {max_iterations} steps): '{task_prompt}'")
        print(f"[COMPUTER AGENT] Display logical resolution: {self.controller.logical_width}x{self.controller.logical_height}")

        summary = "Completed desktop task."
        action_history: List[Tuple[str, str]] = []

        try:
            # Capture initial frame with Set-of-Marks tags
            b64_init, init_w, init_h, elements_summary = self.controller.capture_screen_base64(apply_grid=True, apply_som=True)

            system_prompt = (
                f"You are an expert autonomous macOS desktop GUI automation agent interacting with a logical display resolution of {self.controller.screen_width}x{self.controller.screen_height}.\n\n"
                "=== HIGH-CAPACITY CODE & AUTOMATION OUTPUT CONTRACT ===\n"
                "- COMPLETE & UNABRIDGED IMPLEMENTATIONS: When generating code or text to insert via `computer_type` or write to files via `execute_bash`, you have a generous 4096-token budget. Always output the full, complete, working implementation without truncations, omitted sections, or '# ...rest of code...' placeholders.\n"
                "- DETERMINISTIC HYBRID EXECUTION: Prefer `execute_bash(command=...)` for deterministic CLI or AppleScript operations (e.g. launching applications with `open -a`, writing files, volume/media controls, curl requests) instead of slow visual clicking.\n"
                "- NEVER attempt `sudo` or destructive system modifications.\n\n"
                "=== SET-OF-MARKS (SoM) ACCESSIBILITY GROUNDING ===\n"
                "- Interactive UI buttons, inputs, tabs, search bars, and links are labeled with numeric ID badges: [1], [2], [3]...\n"
                "- ALWAYS PREFER calling `click_element(element_id=...)` with the exact numeric badge for 100% click precision.\n"
                "- Use `computer_click(x, y)` as a fallback only when clicking on an untagged canvas, video player, or custom graphics widget.\n"
                "- Use `computer_hotkey(combo='cmd+t')` or `computer_hotkey(combo='cmd+w')` for fast browser/tab control.\n"
                "- Use `inspect_region(x, y)` if you need a high-resolution zoomed crop of dense code or small text.\n\n"
                "=== MANDATORY STEP-BY-STEP VISUAL VERIFICATION ===\n"
                "- In EVERY tool call, you MUST provide the `observation_and_verification` parameter before selecting the action.\n"
                "- Carefully compare the current screenshot with the previous screenshot. Explicitly state what changed, verify whether the previous action succeeded, and explain why the next action is required.\n"
                "- If an element click or typing action didn't take effect, do not repeat the exact same failed action. Try an alternate approach (e.g. Set-of-Marks ID badge, keyboard shortcut like 'cmd+a' then typing, or inspect_region).\n\n"
                "=== COMPLETION ===\n"
                "- Complete multi-step tasks end-to-end autonomously without stopping for confirmation until calling `finish_task`.\n"
                "- When the user's goal is fully achieved, call finish_task(summary='...') with a concise 1-sentence confirmation suitable for spoken audio."
            )

            init_img_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_init}"
                }
            }

            elements_text = "\n".join([f"  Tag [{e['id']}]: {e['role']} '{e['title']}'" for e in elements_summary[:25]]) if elements_summary else "No accessibility tags detected."

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Task: {task_prompt}\nDetected interactive UI element tags:\n{elements_text}\nInitial screen state:"},
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
                            tools=COMPUTER_ACTION_TOOLS,
                            tool_choice="auto",
                            max_completion_tokens=4096,
                            temperature=0.2,
                        )
                    except Exception as param_err:
                        if "temperature" in str(param_err).lower() or "unsupported_parameter" in str(param_err).lower():
                            response = self.openai_client.chat.completions.create(
                                model=self.computer_use_model,
                                messages=messages,
                                tools=COMPUTER_ACTION_TOOLS,
                                tool_choice="auto",
                                max_completion_tokens=4096,
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
                        b64_next, next_w, next_h, next_elements = self.controller.capture_screen_base64(apply_grid=True, apply_som=True)
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
                    args = _safe_parse_tool_arguments(tool_call.function.arguments or "")

                    # Extract model's visual verification
                    verification = args.pop("observation_and_verification", None)
                    if verification:
                        print(f"[COMPUTER AGENT VERIFICATION] Step {iteration + 1}: {verification}")

                    # Consecutive Action Stall Detection: Check if same mechanical action repeated 3x
                    action_sig = (func_name, json.dumps(args, sort_keys=True))
                    action_history.append(action_sig)

                    if len(action_history) >= 3 and action_history[-1] == action_history[-2] == action_history[-3]:
                        print(f"[COMPUTER AGENT STALL] Detected 3 consecutive identical actions: {action_sig}. Halting loop.")
                        summary = f"Unable to complete task: Action '{func_name}' repeated 3 times with no progress."
                        finished = True
                        break

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
                        "content": f"{tool_result}. Verify visual outcome in next screenshot.",
                    })

                # Perceptual settling check
                self.controller.wait_for_change(timeout_ms=500)

                if finished or self.abort_event.is_set():
                    if self.abort_event.is_set():
                        summary = "Desktop control aborted."
                    print(f"[COMPUTER AGENT] Goal achieved / loop ended on iteration {iteration + 1}!")
                    break

                # Capture updated live screenshot with updated Set-of-Marks tags
                if iteration < max_iterations - 1:
                    b64_next, next_w, next_h, next_elements = self.controller.capture_screen_base64(apply_grid=True, apply_som=True)
                    print(f"[COMPUTER AGENT] Live screenshot updated: {next_w}x{next_h} px ({len(next_elements)} elements tagged)")
                    next_elements_text = "\n".join([f"  Tag [{e['id']}]: {e['role']} '{e['title']}'" for e in next_elements[:25]]) if next_elements else "No accessibility tags detected."
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"[Step {iteration+2}]: Updated live screenshot.\nUpdated interactive UI tags:\n{next_elements_text}\nVerify outcome of previous action and determine next step.",
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
        """Dispatches query to OpenAI gpt-4o-mini / gpt-5.4 with semantic RAG memory and multimodal vision."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

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
                tools=CONVERSATION_TOOLS,
                tool_choice="auto",
                max_completion_tokens=1000,
                temperature=0.2,
            )
        except Exception as err:
            if "temperature" in str(err).lower() or "unsupported_parameter" in str(err).lower():
                response = self.openai_client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    tools=CONVERSATION_TOOLS,
                    tool_choice="auto",
                    max_completion_tokens=1000,
                )
            else:
                raise err

        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            spoken_summary = None

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = _safe_parse_tool_arguments(tool_call.function.arguments or "")

                if func_name == "start_computer_automation":
                    goal = func_args.get("goal", prompt)
                    print(f"[AI] Model initiated start_computer_automation: goal='{goal}'")
                    return self._execute_computer_agent_loop(
                        task_prompt=goal,
                        on_status_change=on_status_change,
                        on_reply_generated=on_reply_generated,
                        max_iterations=30,
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
                            self.on_annotations_generated(anns)
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

            try:
                follow_up = self.openai_client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    max_completion_tokens=250,
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

                if isinstance(func_args, str):
                    func_args = _safe_parse_tool_arguments(func_args)

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
        """Zero-latency local speech synthesis via native macOS say command with lifecycle signals."""
        if not text:
            return
        try:
            self.signals.speaking_started.emit()
            clean_text = text.replace('"', '\\"').replace("'", "’")
            subprocess.run(["say", clean_text])
        except Exception as e:
            print(f"[ERROR] macOS TTS error: {e}")
        finally:
            try:
                self.signals.speaking_finished.emit()
            except Exception:
                pass


# Backward-compatible alias
GeminiAssistant = AIAssistant
