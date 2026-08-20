# 🤖 FRIDAY

**High-Precision Hand & Finger Gesture Desktop Controller for macOS with Dual-Hand Modifier Architecture, Strict Precedence Gating, and Centroid Persistence.**

Control your entire Mac using natural hand and finger gestures from your webcam. Features an intuitive Dual-Hand "Modifier" architecture (where your Left Hand acts as an on-demand physical "Shift Key"), rock-solid velocity-adaptive EMA motion smoothing with a 6px stationary deadzone, spatial hysteresis centroid tracking to eliminate hand swapping when hands are close, strict extension precedence gating so scrolling never conflicts with right-clicks, strict 35px movement gate for Drag & Drop, and native macOS Quartz event dispatch.

---

## ✨ Features

- **Dual-Hand "Modifier" Gesture & Voice Dictation Architecture**:
  - 🖐️ **Right Hand (Dominant)**: Controls sub-pixel cursor movement, primary clicks, two-finger scrolling, and 3-finger space switching.
  - 🖐️ **Left Hand (Physical "Shift Key" Modifier & Voice Dictation Push-to-Talk)**:
    - **Normal Mode (Left Hand Down)**:
      - ☝️ **Point**: Cursor navigation via stable Index MCP Knuckle.
      - 🤏 **Pinch Tap (<250ms)**: Instant **Left Click** at locked $(X,Y)$ target (Middle Finger Curled).
      - ✊ **Pinch Hold (>450ms & >35px move)**: **Drag & Drop** mode.
      - 🤏🤏 **Double Pinch Tap**: **Double Click**.
    - **Modifier Mode (Left Hand Extended)**:
      - 🖐️+✌️ **Two-Finger Scroll (Strict Precedence)**: Extending Index & Middle fingers forces **Scroll Mode Only** (completely bypassing any clicks). Moving hand UP scrolls UP; moving DOWN scrolls DOWN.
      - 🖐️+🤏 **Pinch Tap (<250ms)**: Instant **Right Click** (Context Menu), requiring Middle Finger to be curled.
    - **Push-to-Talk Voice Dictation Mode (Left Hand Fist)**:
      - ✊ **Left Hand Fist**: Activates local speech-to-text powered by `faster-whisper` (int8 quantized). Releasing the fist automatically transcribes your speech and pastes text directly into your active search bar or text field via Cmd+V.

- **Spatial Hysteresis & Centroid Tracking**:
  - Multi-hand identity tracking matches hands frame-to-frame based on Euclidean distance to previous palm centroids.
  - Hands touching, crossing, or moving close to each other in frame will **never swap identities** or drop tracking.

- **Subdued Minimalist Glassmorphic HUD**:
  - **Status Pill**: A compact, non-intrusive acrylic pill floating in the top-right corner (`rgba(24, 24, 27, 0.85)` with 1px soft border) showing live state (`POINTING`, `CLICK`, `DRAGGING`, `RIGHT-CLICK`, `SCROLLING`, `SPACES`, `LISTENING...`), modifier presence (`● [SHIFT]`), and pause toggle.

---

## 🚀 Quick Start (Single Command)

```bash
./run.sh
```

---

## 🖐️ Gesture Cheatsheet

| Gesture | Hand Configuration | Action |
| :--- | :--- | :--- |
| **Move Pointer** | ☝️ Right Index Finger extended | Smooth knuckle-anchored navigation (Index MCP 5) |
| **Left Click** | 🤏 Right Pinch Tap (<250ms, Middle Curled) + Left Down | Clicks at locked $(X,Y)$ target |
| **Drag & Drop** | ✊ Right Pinch Hold >450ms & move >35px (Left Down) | Holds mouse down and drags |
| **Double Click** | 🤏🤏 Double Right Pinch Tap within 400ms | Native macOS double click |
| **Right Click** | 🖐️+🤏 Right Pinch Tap (<250ms, Middle Curled) + **Left Extended** | Context menu / right click |
| **Scroll Up** | 🖐️+✌️ 2 Fingers extended + **Left Extended** (Move UP) | Proportional upward scroll wheel |
| **Scroll Down** | 🖐️+✌️ 2 Fingers extended + **Left Extended** (Move DOWN) | Proportional downward scroll wheel |
| **Space Right** | 🖐️ 3 Fingers extended (Index, Middle, Ring) → Swipe Left | Switch to Desktop Space on Right (`Ctrl+Right`) |
| **Space Left** | 🖐️ 3 Fingers extended (Index, Middle, Ring) → Swipe Right | Switch to Desktop Space on Left (`Ctrl+Left`) |
| **Mission Control** | 🖐️ 3 Fingers extended (Index, Middle, Ring) → Swipe Up | Open/Close Mission Control (`Ctrl+Up`) |
| **Voice Dictation** | ✊ **Left Hand Fist** (Hold while speaking, release to type) | Push-to-Talk speech-to-text via `faster-whisper` |


---

## 🔒 macOS Permissions Setup

1. **Accessibility Permission** *(Required for controlling system cursor & clicks)*:
   - Go to **System Settings** → **Privacy & Security** → **Accessibility**.
   - Enable your terminal or Python executable.
   - *Click "Accessibility" directly inside the FRIDAY HUD drawer to open settings.*

2. **Camera Permission** *(Required for webcam stream)*:
   - Go to **System Settings** → **Privacy & Security** → **Camera**.
   - Ensure Camera access is granted to your terminal / Python app.

---

## 📁 Project Architecture

```
FRIDAY/
├── main.py                # Master application orchestrator & startup diagnostics
├── hud_sidebar.py         # Subdued glassmorphic status pill & settings drawer
├── overlay_window.py      # Transparent click-through monochrome visual reticle
├── vision_engine.py       # Asynchronous CameraGrabber & Multi-Hand Centroid Tracker
├── landmark_smoother.py   # Raw 3D landmark temporal pre-smoothing
├── one_euro_filter.py     # Velocity-Adaptive EMA filter with 6px deadzone
├── gesture_recognizer.py  # Knuckle anchor, strict precedence scroll, and modifier engine
├── mouse_controller.py    # Native macOS Quartz CoreGraphics event injector
├── permissions.py         # macOS Accessibility & Camera permission helper
├── requirements.txt       # Python dependencies
├── run.sh                 # Single-command launcher
└── README.md              # User guide & documentation
```
