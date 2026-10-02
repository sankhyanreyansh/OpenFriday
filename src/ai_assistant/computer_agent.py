"""
Autonomous Set-of-Marks (SoM) Perception-Action Computer Agent for Open FRIDAY.
Executes multi-step desktop workflows using native GUI automation and deterministic CLI tools.
"""

import json
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple

from openai import OpenAI
try:
    from .prompts import build_computer_agent_prompt
    from .tools import COMPUTER_ACTION_TOOLS, _safe_parse_tool_arguments
except ImportError:
    try:
        from ai_assistant.prompts import build_computer_agent_prompt
        from ai_assistant.tools import COMPUTER_ACTION_TOOLS, _safe_parse_tool_arguments
    except ImportError:
        from prompts import build_computer_agent_prompt
        from tools import COMPUTER_ACTION_TOOLS, _safe_parse_tool_arguments

try:
    from src.control.computer_controller import MacComputerController
except ImportError:
    try:
        from control.computer_controller import MacComputerController
    except ImportError:
        from computer_controller import MacComputerController


class ComputerAgent:
    """Orchestrates autonomous desktop perception-action loops with Set-of-Marks precision."""

    def __init__(
        self,
        openai_client: Optional[OpenAI],
        controller: MacComputerController,
        available_tools: Dict[str, Callable],
        abort_event: threading.Event,
        model_id: str = "gpt-4o",
    ):
        self.openai_client = openai_client
        self.controller = controller
        self.available_tools = available_tools
        self.abort_event = abort_event
        self.model_id = model_id
        self.is_controlling_desktop: bool = False

    def execute_loop(
        self,
        task_prompt: str,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_reply_generated: Optional[Callable[[str], None]] = None,
        max_iterations: int = 30,
    ) -> str:
        """
        Autonomous Set-of-Marks (SoM) Perception-Action loop using COMPUTER_ACTION_TOOLS,
        with step-by-step verification, resilient JSON parsing, 4096-token budget, and 3x consecutive identical action stall detection.
        """
        if not self.openai_client:
            fallback_msg = "Computer automation requires an active OpenAI API key."
            if on_reply_generated:
                on_reply_generated(fallback_msg)
            if on_status_change:
                on_status_change("SPEAKING")
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
            cap_init = self.controller.capture_screen_base64(apply_grid=True, apply_som=True)
            b64_init, init_w, init_h = cap_init[0], cap_init[1], cap_init[2]
            elements_summary = cap_init[3] if len(cap_init) > 3 else []
            system_prompt = build_computer_agent_prompt(self.controller.screen_width, self.controller.screen_height)

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
                print(f"[COMPUTER AGENT] Querying {self.model_id} for next UI action...")

                try:
                    try:
                        response = self.openai_client.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
                            tools=COMPUTER_ACTION_TOOLS,
                            tool_choice="auto",
                            max_completion_tokens=4096,
                            temperature=0.2,
                        )
                    except Exception as param_err:
                        if "temperature" in str(param_err).lower() or "unsupported_parameter" in str(param_err).lower():
                            response = self.openai_client.chat.completions.create(
                                model=self.model_id,
                                messages=messages,
                                tools=COMPUTER_ACTION_TOOLS,
                                tool_choice="auto",
                                max_completion_tokens=4096,
                            )
                        else:
                            raise param_err
                except Exception as e:
                    print(f"[COMPUTER AGENT ERROR] Model inference failed at step {iteration+1}: {e}")
                    summary = f"Error during computer agent execution: {e}"
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
                        cap_next = self.controller.capture_screen_base64(apply_grid=True, apply_som=True)
                        b64_next = cap_next[0]
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
                    cap_next = self.controller.capture_screen_base64(apply_grid=True, apply_som=True)
                    b64_next, next_w, next_h = cap_next[0], cap_next[1], cap_next[2]
                    next_elements = cap_next[3] if len(cap_next) > 3 else []
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
        return summary
