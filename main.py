#!/usr/bin/env python3
"""
Open FRIDAY - Root Application Entry Point.
Launches Open FRIDAY from the organized src/ package.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

for sub in ["", "vision", "control", "audio", "ui", "ai_assistant"]:
    p = os.path.join(SRC_DIR, sub) if sub else SRC_DIR
    if p not in sys.path:
        sys.path.insert(0, p)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.main import main

if __name__ == "__main__":
    main()
