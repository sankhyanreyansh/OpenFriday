"""
Open FRIDAY AI Assistant Package.
Provides voice-first conversational intelligence, AR annotations, and Set-of-Marks computer control.
"""

from .assistant import AIAssistant, AIAssistantSignals, GeminiAssistant
from .tools import (
    CONVERSATION_TOOLS,
    COMPUTER_ACTION_TOOLS,
    COMPUTER_USE_TOOLS,
    OPENAI_TOOLS,
    _safe_parse_tool_arguments,
)
from .computer_agent import ComputerAgent
from .prompts import build_system_instruction, build_computer_agent_prompt
from .memory_vault import MemoryVault
from .bash_executor import BashExecutor
from .system_tools import open_website, open_application

__all__ = [
    "AIAssistant",
    "AIAssistantSignals",
    "GeminiAssistant",
    "ComputerAgent",
    "CONVERSATION_TOOLS",
    "COMPUTER_ACTION_TOOLS",
    "COMPUTER_USE_TOOLS",
    "OPENAI_TOOLS",
    "_safe_parse_tool_arguments",
    "build_system_instruction",
    "build_computer_agent_prompt",
    "MemoryVault",
    "BashExecutor",
    "open_website",
    "open_application",
]
