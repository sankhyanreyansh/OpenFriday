"""
Open FRIDAY AI Desktop Assistant Orchestrator.
Coordinates voice-first intent routing, OpenAI API multimodal perception,
semantic RAG memory recall, AR screen annotations, and autonomous desktop control.
"""

import os
import base64
import json
import subprocess
import threading
from typing import Optional, Callable, Dict, Any, List

from PyQt6.QtCore import QObject, pyqtSignal
from dotenv import load_dotenv
from openai import OpenAI

try:
    from .system_tools import open_website, open_application
    from .memory_vault import MemoryVault
    from .bash_executor import BashExecutor
    from .prompts import build_system_instruction
    from .tools import (
        CONVERSATION_TOOLS,
        COMPUTER_ACTION_TOOLS,
        _safe_parse_tool_arguments,
    )
    from .computer_agent import ComputerAgent
except ImportError:
    try:
        from system_tools import open_website, open_application
        from memory_vault import MemoryVault
        from bash_executor import BashExecutor
        from ai_assistant.prompts import build_system_instruction
        from ai_assistant.tools import (
            CONVERSATION_TOOLS,
            COMPUTER_ACTION_TOOLS,
            _safe_parse_tool_arguments,
        )
        from ai_assistant.computer_agent import ComputerAgent
    except ImportError:
        from system_tools import open_website, open_application
        from memory_vault import MemoryVault
        from bash_executor import BashExecutor
        from prompts import build_system_instruction
        from tools import (
            CONVERSATION_TOOLS,
            COMPUTER_ACTION_TOOLS,
            _safe_parse_tool_arguments,
        )
        from computer_agent import ComputerAgent

try:
    from src.control.computer_controller import MacComputerController
except ImportError:
    try:
        from control.computer_controller import MacComputerController
    except ImportError:
        from computer_controller import MacComputerController

load_dotenv()


class AIAssistantSignals(QObject):
    annotations_ready = pyqtSignal(list, str)  # (annotations, spoken_response)
    speaking_started = pyqtSignal()
    speaking_finished = pyqtSignal()


class AIAssistant:
    """
    Open FRIDAY AI Desktop Assistant powered by OpenAI API.
    Provides conversational voice interaction, visual AR screen annotations,
    and autonomous GUI/CLI task execution with Set-of-Marks precision.
    """

    def __init__(
        self,
        model_type: str = "openai",
        model_id: str = "gpt-4o",
        primary_model: str = "gpt-4o-mini",
        computer_use_model: Optional[str] = None,
    ):
        self.model_type = model_type.lower()
        if self.model_type != "openai":
            print(f"[CONFIG NOTICE] Model type '{model_type}' requested, but currently only 'openai' is supported. Using OpenAI API.")

        self.model_id = model_id
        self.primary_model = primary_model
        self._computer_use_model = computer_use_model or model_id
        self.agent_model = self._computer_use_model

        # Native macOS computer controller with Set-of-Marks grounding
        self.controller = MacComputerController()

        # Semantic Vector RAG Memory Vault
        self.memory_vault = MemoryVault()

        # Safe Non-Blocking Bash Executor
        self.bash_executor = BashExecutor(default_timeout=10.0)

        # In-Session Rolling Conversation History
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history_turns: int = 12

        # Abort Event for computer automation
        self.abort_event = threading.Event()

        # Dedicated Qt GUI Signals
        self.signals = AIAssistantSignals()

        # System prompt
        self.system_instruction = build_system_instruction(
            self.controller.screen_width, self.controller.screen_height
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

        # Primary OpenAI Client
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.strip():
            self._openai_client: Optional[OpenAI] = OpenAI(api_key=openai_key.strip())
        else:
            self._openai_client = None
            print("[AI ASSISTANT WARNING] OPENAI_API_KEY not found in environment. Please add it to your .env file.")

        # Autonomous computer agent loop
        self.computer_agent = ComputerAgent(
            openai_client=self._openai_client,
            controller=self.controller,
            available_tools=self.available_tools,
            abort_event=self.abort_event,
            model_id=self._computer_use_model,
        )

        self.context_image_bytes: Optional[bytes] = None
        self.on_context_changed: Optional[Callable[[Optional[bytes]], None]] = None
        self.on_reply_generated: Optional[Callable[[str], None]] = None
        self.on_annotations_generated: Optional[Callable[[List[Dict[str, Any]]], None]] = None

        # Concurrency & busy state locking
        self.is_busy: bool = False
        self.busy_lock = threading.Lock()

    @property
    def openai_client(self) -> Optional[OpenAI]:
        return self._openai_client

    @openai_client.setter
    def openai_client(self, client: Optional[OpenAI]):
        self._openai_client = client
        if hasattr(self, "computer_agent"):
            self.computer_agent.openai_client = client

    @property
    def computer_use_model(self) -> str:
        return self._computer_use_model

    @computer_use_model.setter
    def computer_use_model(self, model: str):
        self._computer_use_model = model
        self.agent_model = model
        if hasattr(self, "computer_agent"):
            self.computer_agent.model_id = model

    @property
    def is_controlling_desktop(self) -> bool:
        if hasattr(self, "computer_agent"):
            return self.computer_agent.is_controlling_desktop
        return getattr(self, "_is_controlling_desktop", False)

    @is_controlling_desktop.setter
    def is_controlling_desktop(self, value: bool):
        self._is_controlling_desktop = value
        if hasattr(self, "computer_agent"):
            self.computer_agent.is_controlling_desktop = value

    def _handle_inspect_region(self, x: int, y: int, radius: int = 150) -> str:
        b64_crop, cw, ch = self.controller.inspect_region(x, y, radius)
        return f"[INSPECT_REGION RESULT: High-resolution {cw}x{ch} px crop around ({x}, {y}) captured successfully.]"

    def clear_history(self):
        """Clears rolling conversational history."""
        self.conversation_history = []

    def run_computer_agent(
        self,
        task_prompt: str,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_reply_generated: Optional[Callable[[str], None]] = None,
        max_iterations: int = 30,
    ) -> str:
        """Runs the autonomous computer agent loop."""
        try:
            summary = self.computer_agent.execute_loop(
                task_prompt=task_prompt,
                on_status_change=on_status_change,
                on_reply_generated=on_reply_generated,
                max_iterations=max_iterations,
            )
            if on_reply_generated:
                try:
                    on_reply_generated(summary)
                except Exception:
                    pass
            if on_status_change:
                on_status_change("SPEAKING")
            self._speak(summary)
            return summary
        finally:
            if on_status_change:
                on_status_change("IDLE")

    def _query_ollama(self, prompt: str, image_bytes: Optional[bytes] = None, **kwargs) -> str:
        """Stub for backward compatibility; Ollama fallback is disabled in favor of OpenAI."""
        if self._is_computer_control_task(prompt):
            return "Computer automation requires an active cloud connection and is unavailable offline."
        return "Ollama fallback has been disabled. Please configure an OpenAI API key."

    def abort_computer_agent(self):
        """Sets abort event to terminate running computer agent immediately."""
        self.abort_event.set()

    def set_context_image(self, image_bytes: bytes):
        """Stores context image (e.g. from screen snip) for next user prompt."""
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
        """Determines if a prompt is asking about visual/screen content."""
        keywords = [
            "screen", "diagram", "circuit", "ui", "look at", "what is this", "what's this",
            "explain this", "where is", "find", "highlight", "annotate", "code on screen",
            "what am i looking at", "on my screen", "what do you see", "show me", "point to",
            "break down", "error on screen", "bug in this", "what does this mean", "describe this",
            "see here", "look here"
        ]
        p_lower = prompt.lower()
        return any(k in p_lower for k in keywords)

    def _is_visual_annotation_query(self, prompt: str) -> bool:
        """Determines if a prompt is asking to explain, point out, annotate, or inspect screen content."""
        return self._is_visual_query(prompt)

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
                            f"- \"target_model\": \"{self.primary_model}\" or \"{self.computer_use_model}\"\n"
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
                if target_model not in (self.primary_model, self.computer_use_model):
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
                print(f"[AI ROUTER WARN] Router call failed ({e}), using heuristic fallback.")

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
                    summary = self.computer_agent.execute_loop(
                        task_prompt=prompt,
                        on_status_change=on_status_change,
                        on_reply_generated=on_reply_generated,
                        max_iterations=30,
                    )
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": summary or "Completed desktop task."})
                    if len(self.conversation_history) > self.max_history_turns * 2:
                        self.conversation_history = self.conversation_history[-self.max_history_turns * 2:]

                    if on_reply_generated:
                        try:
                            on_reply_generated(summary)
                        except Exception:
                            pass
                    if on_status_change:
                        on_status_change("SPEAKING")
                    self._speak(summary)
                    return
                else:
                    fallback_msg = "Computer automation requires an active OpenAI API key."
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
                    snap_res = self.controller.capture_screen_base64(apply_grid=True, apply_som=False)
                    b64_snap = snap_res[0] if isinstance(snap_res, tuple) else snap_res
                    effective_image = base64.b64decode(b64_snap)
                    print(f"[AI] Selective Screen Capture: Acquired live full-screen context for '{prompt}'")
                except Exception as e:
                    print(f"[AI WARN] Selective screen capture error: {e}")

            if on_status_change:
                on_status_change("THINKING")

            target_model = classification.get("target_model", self.primary_model)
            if effective_image is not None:
                target_model = self.computer_use_model

            if not self.openai_client:
                reply = "OpenAI API key not configured. Please add OPENAI_API_KEY to your .env file."
            else:
                reply = self._query_openai(
                    prompt,
                    effective_image,
                    on_annotations_generated=on_annotations_generated,
                    on_status_change=on_status_change,
                    on_reply_generated=on_reply_generated,
                    target_model=target_model,
                    classification=classification,
                )

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
        """Dispatches query to OpenAI with semantic RAG memory and multimodal vision."""
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
                    return self.computer_agent.execute_loop(
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
