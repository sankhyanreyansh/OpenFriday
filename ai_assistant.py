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
                        "description": "Mouse button (default: left)",
                    },
                    "double": {
                        "type": "boolean",
                        "description": "Set to true for double-click (default: false)",
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
            "description": "Type or paste text instantly into the currently focused input field",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The exact string of text to insert/type"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_key",
            "description": "Press a keyboard key or shortcut (e.g. 'enter', 'tab', 'escape', 'space', 'down', 'up', 'cmd+t', 'cmd+w', 'cmd+v', 'cmd+space')",
            "parameters": {
                "type": "object",
                "properties": {
                    "key_name": {"type": "string", "description": "Name of the key or shortcut to press"},
                },
                "required": ["key_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_scroll",
            "description": "Scroll the screen vertically or horizontally",
            "parameters": {
                "type": "object",
                "properties": {
                    "dy": {"type": "integer", "description": "Vertical scroll amount (negative to scroll down, positive to scroll up)"},
                    "dx": {"type": "integer", "description": "Horizontal scroll amount (default: 0)"},
                },
                "required": ["dy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_website",
            "description": "Open a website URL in the default browser",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to open, e.g., 'https://youtube.com'"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Launch or focus a macOS application",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the application, e.g., 'Google Chrome', 'Spotify', 'Notes'"},
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
            "description": "Draw colored visual bounding boxes and callout explanation cards across the user's screen. Use whenever the user asks to explain diagrams, circuits, code, UI, or find elements on screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spoken_response": {
                        "type": "string",
                        "description": "Brief 1-2 sentence spoken explanation for TTS audio."
                    },
                    "annotations": {
                        "type": "array",
                        "description": "List of at least 1 to 5 visual bounding boxes and explanations corresponding to target regions on screen. NEVER return an empty array.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "box_2d": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "[ymin, xmin, ymax, xmax] on a 0-1000 normalized grid."
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Short title (e.g., 'Encoder Block', 'Multi-Head Attention', 'Inputs')."
                                },
                                "text": {
                                    "type": "string",
                                    "description": "1 sentence explanation inside the floating callout card."
                                },
                                "color": {
                                    "type": "string",
                                    "description": "Hex color code (e.g., '#ef4444', '#22c55e', '#3b82f6', '#eab308', '#a855f7')."
                                }
                            },
                            "required": ["box_2d", "label", "text", "color"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["spoken_response", "annotations"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_computer_automation",
            "description": "Trigger when the user asks you to interact with, click, edit, write into, or manipulate any application, editor, browser, or GUI element on their screen.",
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
            "name": "request_user_help",
            "description": "Call this tool if you get stuck, cannot find an element after retrying, encounter a CAPTCHA/2FA, or need the user to position the cursor or focus a specific window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Clear explanation spoken to the user describing what you need them to do."
                    }
                },
                "required": ["explanation"]
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
    """Dual-tier AI assistant client integrating OpenAI API with local Ollama fallback and Computer Use automation."""

    def __init__(
        self,
        primary_model: str = "gpt-4o-mini",
        computer_use_model: str = "gpt-5.4",
        fallback_host: str = "http://127.0.0.1:11434",
    ):
        self.primary_model = primary_model
        self.computer_use_model = computer_use_model
        self.agent_model = computer_use_model
        # Strictly use qwen2.5vl:3b for offline fallback
        self.fallback_model = "qwen2.5vl:3b"
        self.fallback_host = fallback_host

        # Computer GUI controller
        self.controller = MacComputerController()

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
            "You are FRIDAY, an autonomous desktop assistant with full macOS GUI automation (Computer Use) and AR visual annotation capabilities.\n\n"
            f"Screen dimensions: {self.controller.screen_width}x{self.controller.screen_height}.\n\n"
            "DECISION GUIDELINES:\n"
            "1. DESKTOP INTERACTION VS. DIRECT ANSWER:\n"
            "   - If the user asks for code, writing, editing, typing, clicking, or actions relative to something on their screen (e.g., 'write this code below the hello world statement in my editor', 'clear the text in this compiler', 'click on cell B3', 'reply to this message', 'open app and do X'):\n"
            "     -> You MUST call `start_computer_automation` to inspect the screen and execute the typing/clicks directly into their app.\n"
            "   - If the user asks a purely theoretical question, general knowledge, translation, or conversational query with no screen interaction:\n"
            "     -> Answer directly in 1-2 concise sentences for spoken audio.\n"
            "   - If the user asks to visually explain a diagram, architecture, circuit, or find/highlight items on screen:\n"
            "     -> Call `annotate_screen`.\n\n"
            "2. COMPUTER USE RECOVERY:\n"
            "   - Always verify each action in the following screenshot.\n"
            "   - If you struggle to locate an element after 2 attempts or require manual credentials/focus/CAPTCHA, call `request_user_help` to let the user know what assistance is needed rather than looping endlessly.\n\n"
            "3. Spoken Audio: Respond naturally and concisely in 1 to 2 sentences suitable for spoken audio. Never use markdown formatting, asterisks, bullet points, emojis, or code blocks."
        )

        self.available_tools = {
            "open_website": open_website,
            "open_application": open_application,
            "computer_click": self.controller.click,
            "computer_type": self.controller.type_text,
            "computer_key": self.controller.key_press,
            "computer_scroll": self.controller.scroll,
        }
        self.tools = [open_website, open_application]

        self.context_image_bytes: Optional[bytes] = None
        self.on_context_changed: Optional[Callable[[Optional[bytes]], None]] = None
        self.on_reply_generated: Optional[Callable[[str], None]] = None
        self.on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None

        # Concurrency & busy state locking
        self.is_busy: bool = False
        self.busy_lock = threading.Lock()

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

    def query(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_reply_generated: Optional[Callable[[str], None]] = None,
        on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ):
        """Sends prompt (and optional image) in a background thread and speaks the response."""
        if not prompt or not prompt.strip():
            return

        with self.busy_lock:
            if self.is_busy:
                print(f"[AI] Assistant is currently busy. Discarding overlapping query: '{prompt.strip()}'")
                return
            self.is_busy = True

        effective_image = image_bytes if image_bytes is not None else self.context_image_bytes

        # Universal screen perception: always capture live full-screen context if no image is staged
        if effective_image is None:
            try:
                b64_snap, _, _ = self.controller.capture_screen_base64()
                effective_image = base64.b64decode(b64_snap)
                print(f"[AI] Universal screen awareness: Captured live full-screen context for '{prompt.strip()}'")
            except Exception as e:
                print(f"[AI WARN] Automatic screen capture error: {e}")

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
            # Check if this should run as an autonomous computer agent loop directly
            if self._is_computer_control_task(prompt):
                if self.openai_client is not None:
                    self._execute_computer_agent_loop(prompt, on_status_change, on_reply_generated)
                    return
                else:
                    # Offline fallback guardrail
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

            if on_status_change:
                on_status_change("THINKING")

            reply: Optional[str] = None

            # 1. Attempt Primary OpenAI API Dispatch
            if self.openai_client is not None:
                try:
                    reply = self._query_openai(
                        prompt,
                        image_bytes,
                        on_annotations_generated,
                        on_status_change=on_status_change,
                        on_reply_generated=on_reply_generated,
                    )
                except Exception as e:
                    print(f"[AI] OpenAI unavailable ({e}), falling back to local Qwen 3B...")
                    reply = None

            # 2. Fallback to Local Ollama Engine (Strict Qwen 3B)
            if reply is None:
                reply = self._query_ollama(prompt, image_bytes)

            if not reply or not reply.strip():
                reply = "I didn't receive a response."

            clean_reply = reply.strip()

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

        if on_status_change:
            on_status_change("CONTROLLING")

        print(f"\n[COMPUTER AGENT] Starting desktop GUI automation task: '{task_prompt}'")
        print(f"[COMPUTER AGENT] Display logical resolution: {self.controller.logical_width}x{self.controller.logical_height}")

        # Capture initial frame to anchor initial perception
        b64_init, init_w, init_h = self.controller.capture_screen_base64()

        system_prompt = (
            f"You are an expert desktop GUI automation agent interacting with a macOS screen of logical resolution {self.controller.screen_width}x{self.controller.screen_height}.\n\n"
            "COORDINATE SYSTEM:\n"
            "- Provide all coordinates (x, y) on a normalized 0 to 1000 scale:\n"
            "  * (0, 0) = Top-Left corner\n"
            "  * (500, 500) = Exact center of the screen\n"
            "  * (1000, 1000) = Bottom-Right corner\n"
            "  * Typical browser search/address bars are located horizontally centered near the top: (500, 75) to (500, 120).\n\n"
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

        summary = "Completed desktop task."

        for iteration in range(max_iterations):
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
                elif func_name == "request_user_help":
                    explanation = args.get("explanation", "I need your assistance to proceed.")
                    print(f"[COMPUTER AGENT] request_user_help triggered: {explanation}")
                    summary = explanation
                    finished = True
                    tool_result = f"Paused computer agent to request user help: {explanation}"
                    if on_status_change:
                        on_status_change("WAITING")
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

            if finished:
                print(f"[COMPUTER AGENT] Goal achieved / paused on iteration {iteration + 1}!")
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

        if on_status_change and not (func_name == "request_user_help" if 'func_name' in locals() else False):
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
    ) -> str:
        """Dispatches query to OpenAI gpt-4o-mini / gpt-5.4 with support for function tools and multimodal vision."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_instruction}
        ]

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

        target_model = self.computer_use_model if image_bytes else self.primary_model
        print(f"[AI] Dispatching query to {target_model} (multimodal={image_bytes is not None})...")

        try:
            response = self.openai_client.chat.completions.create(
                model=target_model,
                messages=messages,
                tools=COMPUTER_USE_TOOLS,
                tool_choice="auto",
                max_completion_tokens=400,
                temperature=0.2,
            )
        except Exception as err:
            if "temperature" in str(err).lower() or "unsupported_parameter" in str(err).lower():
                response = self.openai_client.chat.completions.create(
                    model=target_model,
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
                elif func_name == "request_user_help":
                    explanation = func_args.get("explanation", "I need your assistance to proceed.")
                    if on_status_change:
                        on_status_change("WAITING")
                    return explanation
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

            # Follow-up completion for natural spoken confirmation
            try:
                follow_up = self.openai_client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    max_completion_tokens=100,
                    temperature=0.2,
                )
                return follow_up.choices[0].message.content or ""
            except Exception:
                return "Completed request."
        else:
            return msg.content or ""

    def _query_ollama(self, prompt: str, image_bytes: Optional[bytes]) -> str:
        """Dispatches query strictly to local Ollama qwen2.5vl:3b as offline fallback."""
        # Fallback guardrail: check if computer automation was requested
        if self._is_computer_control_task(prompt):
            return "Computer automation requires an active cloud connection and is unavailable offline."

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

