"""
Deterministic macOS System Tools for FRIDAY AI Assistant.
Provides safe, native system actions for launching applications and opening websites.
"""

import subprocess
import webbrowser
from typing import Dict

APP_ALIASES: Dict[str, str] = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "safari": "Safari",
    "finder": "Finder",
    "notes": "Notes",
    "apple notes": "Notes",
    "spotify": "Spotify",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm",
    "calculator": "Calculator",
    "calendar": "Calendar",
    "settings": "System Settings",
    "system settings": "System Settings",
    "system preferences": "System Settings",
    "messages": "Messages",
    "mail": "Mail",
    "music": "Music",
    "photos": "Photos",
    "reminders": "Reminders",
    "maps": "Maps",
    "code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
    "slack": "Slack",
    "discord": "Discord",
    "notion": "Notion",
    "cursor": "Cursor",
    "preview": "Preview",
    "textedit": "TextEdit",
    "activity monitor": "Activity Monitor",
}


def open_website(url: str) -> str:
    """Opens a website in the default macOS web browser.

    Args:
        url: The web URL or domain name to open, e.g. 'https://youtube.com', 'github.com', 'google.com'.
    """
    clean_url = url.strip()
    if not clean_url:
        return "No URL provided."

    # If domain only without protocol or TLD, normalize common domains
    if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
        if "." not in clean_url:
            clean_url = f"{clean_url}.com"
        clean_url = f"https://{clean_url}"

    try:
        webbrowser.open(clean_url)
        return f"Opened website: {clean_url}"
    except Exception as e:
        # Fallback to macOS open command
        try:
            subprocess.run(["open", clean_url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Opened website: {clean_url}"
        except Exception as e2:
            return f"Failed to open website {clean_url}: {e2}"


def open_application(app_name: str) -> str:
    """Launches or switches to a macOS application installed on the system.

    Args:
        app_name: The name of the macOS application to launch, e.g. 'Google Chrome', 'Notes', 'Spotify', 'Finder'.
    """
    clean_name = app_name.strip()
    if not clean_name:
        return "No application name provided."

    target_app = APP_ALIASES.get(clean_name.lower(), clean_name)

    try:
        res = subprocess.run(
            ["open", "-a", target_app],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            return f"Launched application: {target_app}"
        else:
            # Try raw input if alias mapping was not installed
            if target_app != clean_name:
                res2 = subprocess.run(["open", "-a", clean_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res2.returncode == 0:
                    return f"Launched application: {clean_name}"

            err_msg = res.stderr.strip() or f"Application '{target_app}' not found."
            return f"Could not launch application '{target_app}': {err_msg}"
    except Exception as e:
        return f"Error launching application '{target_app}': {e}"
