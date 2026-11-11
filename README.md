# Open FRIDAY 🚀

**Open FRIDAY** is a low-latency, touchless hand-gesture desktop controller engineered natively for macOS. Using your MacBook or external webcam, FRIDAY tracks your hands at 60 FPS using MediaPipe Tasks and CoreGraphics event taps to control your cursor, clicks, scrolling, desktop spaces, and window shortcuts without touching the trackpad or keyboard.

---

## 🖐️ Gesture Control Reference

FRIDAY uses a **Dual-Hand Asymmetric Architecture**:
* **Right Hand (Dominant)**: Drives continuous cursor navigation, pointer clicks, instant dragging, and spatial navigation.
* **Left Hand (Modifier / Shift)**: Acts as a physical modifier key (similar to Shift/Fn) to unlock secondary actions like scrolling, right-clicking, and dual-hand zooming.

| Gesture | Hand(s) | Action | macOS Equivalent |
| :--- | :--- | :--- | :--- |
| **Point (Index Extended)** | Right Hand | Move cursor across screen | Mouse movement |
| **Pinch Tap (<350ms)** | Right Hand | Left Click | Left click |
| **Double Pinch Tap** | Right Hand | Double Click | Double click |
| **Pinch & Move (>28px)** | Right Hand | Click & Drag | Mouse drag |
| **Left Hand Raised + Pinch** | Both Hands | Right Click | Secondary click |
| **Left Hand Raised + 2 Fingers** | Both Hands | Continuous Inertial Scroll | Two-finger scroll |
| **Dual Pinch + Spread/Pinch** | Both Hands | Spatial Zoom In / Out | `⌘ +` / `⌘ -` |
| **Clenched Left Fist Alone (Hold)** | Left Hand Alone | Hold-to-Talk Speech Dictation | Record while held, paste on release |
| **3-Finger Flick Left / Right** | Right Hand | Switch Desktop Spaces | `Ctrl + ➔` / `Ctrl + ⬅` |
| **3-Finger Flick Up / Down** | Right Hand | Mission Control / App Exposé | `Ctrl + ⬆` / `Ctrl + ⬇` |
| **Dual Open Palms (Hold 250ms)**| Both Hands | Summon Radial Shortcut Wheel | Modal shortcut menu |

---

## ⚡ Architecture & Performance Highlights

* **MediaPipe Video Tracking**: Operates in `RunningMode.VIDEO`, reusing hand bounding boxes across frames for 20–40% lower CPU usage and higher landmark stability.
* **Native CoreGraphics & Ordered System Events**: Shortcuts (Radial menu, Zoom) dispatch via Quartz `CGEventPost` with sub-millisecond latency. Space switching and Mission Control execute via dedicated FIFO-queued macOS System Events via AppleScript, as macOS WindowServer intentionally filters out synthetic CGEvents for Dock/Spaces hotkeys.
* **Zero-Delay Mouse Events**: Left clicks, double clicks, and right clicks use native Quartz `kCGMouseEventClickState` flags with monotonic boot timestamps (`time.monotonic_ns()`) and without blocking sleeps, guaranteeing 0 dropped video frames.
* **Apex Inflection Swipe Engine**: Rapid successive 3-finger space flicks (e.g. Left $\rightarrow$ Left $\rightarrow$ Left) are evaluated from the motion turnaround point (apex), completely eliminating history dilution while suppressing opposite recoil bounce-back.
* **1€ Dynamic Velocity Filtering**: Adaptive cutoff filtering ensures rock-solid stationary pointing jitter reduction with zero lag during high-speed flicks.
* **Glassmorphic Status Pill**: Real-time status indicator floating in the top right of the desktop with dirty-state styling optimizations to avoid Qt stylesheet re-parsing.
* **Local Whisper Voice Dictation**: Raise clenched left fist alone to dictate; speak for as long as your fist is held up, and drop your hand to automatically transcribe locally via `faster-whisper` (INT8) and paste directly into any active input box with trailing space and automatic clipboard restoration.

---

## ⚙️ System Requirements & Permissions

* **OS**: macOS Sonoma (14.0+) or macOS Sequoia (15.0+)
* **Python**: 3.10+ (tested with Python 3.14)
* **Hardware**: Apple Silicon (M1/M2/M3/M4/M5) or Intel Mac with built-in or USB webcam

### macOS Permissions
FRIDAY requires two standard macOS permissions:
1. **Camera**: To capture hand video frames for MediaPipe landmark tracking.
2. **Accessibility**: To synthesize native cursor movements, clicks, and window navigation events (`System Settings -> Privacy & Security -> Accessibility`).

---

## 🚀 Getting Started

Launch Open FRIDAY using the launcher script:

```bash
./run.sh
```

The script will automatically detect or configure the local Python virtual environment (`.venv`), install dependencies, and launch the application.

### Command-Line Arguments
```bash
./run.sh --help
./run.sh --camera 0          # Select camera device index (default: 0)
./run.sh --no-overlay        # Disable transparent on-screen reticle overlay
./run.sh --no-sidebar        # Disable floating status pill
```

---

## 🎛️ Customization

### 1. Radial Shortcut Wheel
To customize the shortcuts in the radial wheel, edit `RADIAL_ACTIONS` in [`src/control/radial_menu.py`](src/control/radial_menu.py):

```python
RADIAL_ACTIONS = [
    {"id": "COPY",      "label": "COPY",      "symbol": "⌘C", "key_code": 8,  "flags": "command"},
    {"id": "PASTE",     "label": "PASTE",     "symbol": "⌘V", "key_code": 9,  "flags": "command"},
    {"id": "BACKSPACE", "label": "BACKSPACE", "symbol": "⌫",  "key_code": 51, "flags": None},
    {"id": "ENTER",     "label": "ENTER",     "symbol": "↵",  "key_code": 36, "flags": None},
    {"id": "ESCAPE",    "label": "ESCAPE",    "symbol": "⎋",  "key_code": 53, "flags": None},
    {"id": "NEW_TAB",   "label": "NEW TAB",   "symbol": "⌘T", "key_code": 17, "flags": "command"},
    {"id": "CLOSE_TAB", "label": "CLOSE TAB", "symbol": "⌘W", "key_code": 13, "flags": "command"},
]
```
The UI overlay dynamically calculates slice angles and hit detection for any number of actions.

### 2. Sensitivity & Tracking Configuration
Settings can be customized in [`config.py`](config.py) or overridden in `config.json`:
* `pinch_start_threshold`: Normalized thumb-index distance to initiate click/drag (default: `0.38`).
* `pinch_release_threshold`: Distance to release click/drag (default: `0.52`).
* `min_cutoff` / `beta`: One-Euro filter jitter suppression and velocity adaptation coefficients.
* `scroll_sensitivity`: Multiplier for vertical scrolling speed.

---

## 🧪 Running Tests

A comprehensive unit test suite is available under `tests/`:

```bash
.venv/bin/pytest tests/
```
