#!/usr/bin/env bash
# ==============================================================================
# Open FRIDAY - macOS Hand Gesture Desktop Controller Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect existing virtual environment (.venv or .venv311)
if [ -d "$SCRIPT_DIR/.venv" ]; then
    VENV_PATH="$SCRIPT_DIR/.venv"
elif [ -d "$SCRIPT_DIR/.venv311" ]; then
    VENV_PATH="$SCRIPT_DIR/.venv311"
else
    VENV_PATH="$SCRIPT_DIR/.venv"
    echo "Creating Python virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
    echo "Installing required dependencies..."
    "$VENV_PATH/bin/pip" install --upgrade pip
    "$VENV_PATH/bin/pip" install -r requirements.txt
fi


# Ensure all dependencies are satisfied
if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "❌ Error: Python binary not found in $VENV_PATH"
    exit 1
fi

export PYTHONUNBUFFERED=1
exec "$VENV_PATH/bin/python" "$SCRIPT_DIR/main.py" "$@"


