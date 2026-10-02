"""
Centralized Configuration Manager for Open FRIDAY.
Maintains system defaults for tracking, vision, and mouse control in code,
while loading model preferences from config.json.
"""

import sys
import os
import json
from typing import Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for sub in ["", "vision", "control", "audio", "ui", "ai_assistant"]:
    p = os.path.join(SRC_DIR, sub) if sub else SRC_DIR
    if p not in sys.path:
        sys.path.insert(0, p)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    # Application & AI Model Settings
    "app_name": "Open FRIDAY",
    "model_type": "openai",
    "model_id": "gpt-4o",

    # Computer Vision & Tracking Settings
    "tracking_enabled": True,
    "mouse_control_enabled": True,
    "mirror_horizontal": True,
    "camera_id": 0,
    "margin_x": 0.16,
    "margin_y": 0.16,

    # Dynamic Filter & Sensitivity Settings
    "min_cutoff": 0.2,
    "beta": 0.003,
    "pinch_threshold": 0.4,
    "scroll_sensitivity": 1.0,

    # HUD Overlays
    "show_reticle": True,
    "show_pinch_meter": True,
}


def load_config(config_path: str = CONFIG_FILE_PATH) -> Dict[str, Any]:
    """
    Loads configuration settings, merging defaults with user overrides from config.json.
    Missing keys in config.json automatically fall back to DEFAULT_CONFIG values.
    """
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
        except Exception as e:
            print(f"[CONFIG WARNING] Failed to parse {config_path}: {e}. Using defaults.")
    return cfg
