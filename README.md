# OpenFriday

OpenFriday is a macOS desktop controller that uses a webcam and computer vision to control cursor movement, mouse clicks, scrolling, desktop spaces, keyboard shortcuts, and voice dictation without touching a mouse or keyboard.

---

## What It Does

OpenFriday uses your right hand for cursor navigation and your left hand as a modifier:

| Gesture | Hand(s) | Action | macOS Shortcut |
| :--- | :--- | :--- | :--- |
| **Point (Index extended)** | Right Hand | Move cursor | Mouse movement |
| **Pinch tap** | Right Hand | Left Click | Left click |
| **Double pinch tap** | Right Hand | Double Click | Double click |
| **Pinch and drag (>28px)** | Right Hand | Click and Drag | Mouse drag |
| **Left hand up + Pinch** | Both Hands | Right Click | Secondary click |
| **Left hand up + 2 fingers** | Both Hands | Continuous Scroll | Two-finger scroll |
| **Dual pinch + spread/close** | Both Hands | Zoom In / Out | `Cmd +` / `Cmd -` |
| **Clenched left fist (hold)** | Left Hand alone | Voice Dictation | Transcribes speech and pastes |
| **3-finger flick left / right** | Right Hand | Switch Spaces | `Ctrl + Right` / `Ctrl + Left` |
| **3-finger flick up / down** | Right Hand | Mission Control / App Expose | `Ctrl + Up` / `Ctrl + Down` |
| **Dual open palms (hold 250ms)** | Both Hands | Radial Menu | Shortcuts wheel |

### Radial Menu Shortcuts
Opening both palms displays a 7-sector shortcut wheel centered on the screen. Selecting a sector with a pinch tap executes:
- **Copy** (`Cmd + C`)
- **Paste** (`Cmd + V`)
- **Backspace** (`Delete`)
- **Enter** (`Return`)
- **Escape** (`Escape`)
- **New Tab** (`Cmd + T`)
- **Close Tab** (`Cmd + W`)

Custom shortcuts can be configured in [`src/control/radial_menu.py`](src/control/radial_menu.py).

### Voice Dictation
Raising your left hand in a closed fist activates speech dictation. Audio is recorded while the fist is held up, transcribed locally via `faster-whisper`, pasted with a trailing space at the cursor location, and your previous clipboard contents are restored.

---

## Installation

### Prerequisites
- macOS Sonoma (14.0+) or macOS Sequoia (15.0+)
- Python 3.10+
- Built-in or USB webcam

### Permissions
macOS requires two permissions:
1. **Camera**: Required for hand tracking (`System Settings -> Privacy & Security -> Camera`).
2. **Accessibility**: Required to generate mouse and keyboard events (`System Settings -> Privacy & Security -> Accessibility`).

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/reyanshsankhyan/OpenFriday.git
   cd OpenFriday
   ```

2. Run the automated launcher:
   ```bash
   ./run.sh
   ```
   `run.sh` creates a virtual environment in `.venv/`, installs dependencies from `requirements.txt`, downloads the hand landmark model if missing, and launches the application.

   Or install manually:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python main.py
   ```

---

## Usage and Flags

```bash
./run.sh [options]
```

| Flag | Description |
| :--- | :--- |
| `--camera <id>` | Camera device index (default: `0`) |
| `--no-overlay` | Disable on-screen reticle overlay |
| `--no-sidebar` | Disable floating status pill |

---

## Configuration

Tracking thresholds, smoothing filters, and input devices can be adjusted in [`config.py`](config.py) or overridden in `config.json`:

- `pinch_start_threshold`: Threshold to trigger pinch (default: `0.38`).
- `pinch_release_threshold`: Threshold to release pinch (default: `0.52`).
- `scroll_sensitivity`: Speed multiplier for scrolling (default: `1.0`).
- `mic_device`: Audio input device index or `null` for system default.

---

## Running Tests

Run the test suite with pytest:

```bash
.venv/bin/pytest tests/
```
