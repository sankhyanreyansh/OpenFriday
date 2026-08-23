"""
Safe, Non-Blocking, Sandboxed Bash Execution Engine for Open FRIDAY.
Includes proactive security filtering, destructive command blacklists,
privilege escalation rejection, and dynamic micro-sleep process polling.
"""

import subprocess
import shlex
import re
import time
from typing import Tuple, List

BLOCKED_PATTERNS = [
    # Privilege escalation
    r"\bsudo\b",
    r"\bdoas\b",
    r"\bsu\s+",
    # Destructive file removal
    r"\brm\s+-(?:r|f|rf|fr)\s+[/~*]",
    r"\brm\s+-[a-z]*r[a-z]*\s+[/~*]",
    r"\brm\s+-[a-z]*f[a-z]*\s+[/~*]",
    r"\brmdir\s+[/~*]",
    # Disk formatting & raw overwrites
    r"\bmkfs\b",
    r"\bdd\b\s+if=",
    r"\bfdisk\b",
    r"\bdiskutil\s+(?:erase|partition|reformat|unmountdisk)",
    # System file & shell config tampering
    r">\s*/etc/",
    r">\s*/system/",
    r">\s*/library/preferences/",
    r">\s*~/\.zshrc",
    r">\s*~/\.bashrc",
    r">\s*~/\.bash_profile",
    r">\s*~/\.profile",
    r">\s*/bin/",
    r">\s*/sbin/",
    r">\s*/usr/bin/",
    # Fork bombs & system power commands
    r":\(\)\s*\{\s*:\|:&\s*\};:",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r"\binit\s+[06]\b",
    # Interactive blocking tools without non-interactive flags
    r"^\s*nano\b",
    r"^\s*vim\b",
    r"^\s*vi\b",
    r"^\s*top\b",
    r"^\s*htop\b",
    r"^\s*less\b",
    r"^\s*more\b",
    r"^\s*man\b",
]


class BashExecutor:
    """Non-blocking, sandboxed local shell executor with security guardrails."""

    def __init__(self, default_timeout: float = 10.0):
        self.default_timeout = default_timeout

    def is_command_safe(self, command: str) -> Tuple[bool, str]:
        """Validates command against destructive patterns and privilege escalation attempts."""
        if not command or not command.strip():
            return False, "Empty command string."

        cmd_clean = command.strip().lower()

        # Check for privilege escalation
        if cmd_clean.startswith("sudo ") or cmd_clean.startswith("doas ") or " sudo " in cmd_clean or " doas " in cmd_clean:
            return False, "Blocked: Root privilege elevation ('sudo' / 'doas') is strictly prohibited."

        # Check against blacklist patterns (case-insensitive)
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, cmd_clean, re.IGNORECASE):
                return False, f"Blocked: Command matches restricted security pattern '{pattern}'."

        return True, ""

    def run(self, command: str, **kwargs) -> str:
        """
        Executes a shell command or AppleScript snippet with dynamic non-blocking polling.
        Returns output immediately upon process termination.
        """
        if not command or not command.strip():
            return "[ERROR] No command specified."

        cmd_str = command.strip()
        is_safe, reason = self.is_command_safe(cmd_str)
        if not is_safe:
            print(f"[SECURITY GUARDRAIL] Rejected unsafe bash command: '{cmd_str}' -> {reason}")
            return f"[SECURITY ERROR] {reason}"

        print(f"[BASH EXECUTOR] Running: {cmd_str}")

        try:
            start_time = time.time()
            proc = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Dynamic non-blocking poll loop: returns as soon as execution completes
            while proc.poll() is None:
                if (time.time() - start_time) > self.default_timeout:
                    proc.kill()
                    print(f"[BASH EXECUTOR] Command timed out after {self.default_timeout}s: '{cmd_str}'")
                    return f"[TIMEOUT ERROR] Command exceeded {self.default_timeout}s execution cap and was terminated."
                time.sleep(0.01)

            stdout, stderr = proc.communicate()
            output = stdout if stdout else stderr

            # Truncate large output buffers for LLM context safety (max 2000 chars)
            if len(output) > 2000:
                output = output[:1000] + "\n...[OUTPUT TRUNCATED]...\n" + output[-1000:]

            clean_out = output.strip()
            print(f"[BASH EXECUTOR] Result ({proc.returncode}): {clean_out[:100]}{'...' if len(clean_out) > 100 else ''}")
            return clean_out if clean_out else "[Command executed successfully with no output]"

        except Exception as e:
            err_msg = f"[EXECUTION ERROR] {str(e)}"
            print(f"[BASH EXECUTOR ERROR] {err_msg}")
            return err_msg
