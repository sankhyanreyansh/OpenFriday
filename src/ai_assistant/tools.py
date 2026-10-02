"""
Tool definitions and robust argument parsing for Open FRIDAY AI Assistant.
"""

import json
import re
from typing import Dict, Any, List


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
    for field in ["element_id", "x", "y", "delta_y", "radius", "start_x", "start_y", "end_x", "end_y"]:
        num_match = re.search(rf'"{field}"\s*:\s*(\d+)', raw_args)
        if num_match:
            result[field] = int(num_match.group(1))

    # Extract string fields
    for field in ["combo", "url", "app_name", "summary", "button", "key", "goal", "command", "category_file", "fact_or_preference"]:
        str_match = re.search(rf'"{field}"\s*:\s*"([^"]+)"', raw_args)
        if str_match:
            result[field] = str_match.group(1)

    # Extract boolean fields
    for field in ["press_enter", "double_click"]:
        bool_match = re.search(rf'"{field}"\s*:\s*(true|false)', raw_args, re.IGNORECASE)
        if bool_match:
            result[field] = bool_match.group(1).lower() == "true"

    return result


# Conversation-Level Tools (Dispatched during standard queries & routing)
CONVERSATION_TOOLS: List[Dict[str, Any]] = [
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
COMPUTER_ACTION_TOOLS: List[Dict[str, Any]] = [
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
