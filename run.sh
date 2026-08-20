#!/usr/bin/env bash
# ==============================================================================
# FRIDAY - macOS Hand Gesture Desktop Controller Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="$SCRIPT_DIR/.venv311"

echo "=================================================================="
echo "🤖  FRIDAY - macOS Hand Gesture Desktop Controller"
echo "=================================================================="

# Check if .venv311 exists, if not, create it
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating Python virtual environment..."
    if command -v conda &> /dev/null; then
        conda create -y -p "$VENV_PATH" python=3.11
    else
        python3 -m venv "$VENV_PATH"
    fi
    echo "Installing required dependencies..."
    "$VENV_PATH/bin/pip" install --upgrade pip
    "$VENV_PATH/bin/pip" install -r requirements.txt
fi

# Ensure all dependencies are satisfied
if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "❌ Error: Python binary not found in $VENV_PATH"
    exit 1
fi

echo "Starting FRIDAY..."
export PYTHONUNBUFFERED=1
exec "$VENV_PATH/bin/python" "$SCRIPT_DIR/main.py" "$@"
