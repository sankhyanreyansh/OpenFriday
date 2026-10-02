# Open FRIDAY: Intelligent Gesture Controller and Desktop Assistant for macOS

Open FRIDAY is an advanced, touchless desktop assistant for macOS that seamlessly integrates webcam computer vision gestures, ultra-fast local voice dictation, conversational intelligence, visual AR annotations, and autonomous desktop computer control.

By tracking hand landmarks through any standard webcam and listening for voice commands, Open FRIDAY allows you to navigate macOS, trigger productivity shortcuts, dictate text directly into active apps, inspect on-screen content with visual AR callouts, and safely delegate complex multi-step workflows to an autonomous desktop agent.

---

## Core Capabilities

### 1. Touchless Hand Gesture Navigation
- **Sub-Pixel Cursor Control**: Point and glide naturally across your desktop using your right index finger with 1-Euro smoothing filters for jitter-free tracking.
- **Clicking and Selection**: Quick pinch taps for left clicks, double-tap pinches for double clicks, and held pinches for drag-and-drop operations.
- **Inertial Scrolling**: Raise your left open hand and extend your right index and middle fingers to scroll vertically with natural inertia and momentum.
- **Spatial Pinch Zoom**: Pinch with both hands simultaneously and stretch apart to zoom in, or compress together to zoom out.
- **Workspace Navigation**: Swipe three fingers horizontally to switch between macOS Spaces, or swipe upward to toggle Mission Control.
- **Radial Shortcut Wheel**: Flash both open palms to summon a circular HUD menu with instant shortcuts for Enter, New Tab, Close Tab, and Escape.
- **Screen Snipping**: Raise your left hand and drag a bounding box with your right pinch to capture a targeted screen snippet for instant AI analysis.

### 2. Voice Intelligence & Smart Dictation Routing
- **Push-to-Talk Gesture**: Hold a closed fist with your **left hand** in camera view to activate listening mode. Release the fist to complete recording and initiate transcription via local `faster-whisper`.
- **Intelligent Wake-Word Routing**:
  - **AI Assistant Queries**: Begin speaking with **"Friday"** or **"Hey Friday"** (e.g., *"Friday, open Safari and search for recent AI papers"* or *"Friday, explain this diagram on my screen"*). The assistant responds with voice output and executes required actions.
  - **Direct Desktop Dictation**: Speak naturally **without** the wake word (e.g., composing an email or writing a search query). Open FRIDAY automatically pastes your transcribed text directly into the focused application via `Cmd+V`.
- **Visual AR Screen Annotations**: When you ask about code, UI elements, charts, or diagrams on your screen, Open FRIDAY projects high-visibility HUD bounding boxes and callout labels directly over your active display without hijacking mouse control.

### 3. Autonomous Computer Control & Sandboxed Execution
- **Hybrid Desktop Automation**: Open FRIDAY combines macOS GUI interactions (mouse movement, clicks, typing, keyboard shortcuts) with a high-speed, sandboxed Bash tool (`execute_bash`) for deterministic commands (`open -a`, AppleScripts, CLI workflows).
- **Hardened Execution Sandbox**: Non-blocking subprocess execution with real-time stream monitoring and strict security safeguards:
  - Immediate rejection of `sudo`, `doas`, or privilege-escalation commands.
  - Blacklist protection against destructive operations (`rm -rf`, disk partitioning, formatting, system configuration tampering).
- **Self-Verification & Anti-Stall Loop Guard**:
  - Every autonomous iteration requires a step-by-step verification block assessing changes between screenshots to eliminate visual hallucinations.
  - Automatic stall detection breaks execution and alerts the user if the model emits 3 consecutive identical actions or coordinates.
  - Extended 30-step execution horizon for multi-step desktop tasks.
- **Safety Abort Gesture**: Cross your hands, wrists, or fingers into an **"X" shape** at any moment during autonomous computer use to immediately terminate execution and regain full manual control.

### 4. Semantic RAG Memory Vault
- **Local Embedding Vector Search**: Powered by `fastembed` (`bge-small-en-v1.5`) running entirely offline on your Mac. No memory vectors or personal notes are uploaded to third-party embedding APIs.
- **Context-Aware Automatic Recall**: Automatically indexes personal notes, workflow preferences, and project details in `memory_vault/` and injects relevant knowledge chunks into conversations when queried.
- **Human-Readable Storage**: Memory files are standard Markdown files (`user_profile.md`, `projects.md`, `notes.md`) that you can inspect or modify in any text editor.

---

## Gesture Reference

| Action | Hand Configuration | Description |
| :--- | :--- | :--- |
| **Move Pointer** | Right index finger extended | Moves the system mouse pointer across displays. |
| **Left Click** | Right pinch tap (thumb + index) | Executes an immediate left click. |
| **Double Click** | Double right pinch tap within 0.4s | Executes a double click. |
| **Drag & Drop** | Right pinch hold and move | Holds mouse down to select text or drag window elements. |
| **Right Click** | Left hand open + Right pinch tap | Opens context menus or secondary actions. |
| **Scroll Up / Down** | Left hand open + Right two fingers extended | Move hand up to scroll up; move down to scroll down. |
| **Spatial Zoom** | Both hands pinched (stretch / compress) | Stretch apart to zoom in; bring together to zoom out. |
| **Next / Prev Space** | Right three fingers extended (swipe left/right) | Switches macOS desktop spaces. |
| **Mission Control** | Right three fingers extended (swipe up) | Toggles macOS Mission Control. |
| **Screen Snippet** | Left hand open + Right pinch drag | Draws a crop area on screen for targeted AI queries. |
| **Push-to-Talk** | Left hand closed fist (hold while speaking) | Activates voice recording; releases to transcribe. |
| **Radial Shortcut Menu** | Both hands open palms for 0.3s | Displays on-screen HUD wheel (Enter, Tab, Close, Esc). |
| **Safety Abort** | Crossed hands, wrists, or fingers ("X" shape) | Instantly aborts autonomous computer automation loops. |

---

## Local Setup Guide for macOS

### Prerequisites
- macOS Monterey (12.0) or newer (optimized for Apple Silicon M1/M2/M3/M4 and Intel Macs).
- Built-in FaceTime HD camera or external USB webcam.
- Python 3.10 through 3.14.

---

### Step 1: Clone the Repository
Open Terminal and navigate to your workspace:

```bash
git clone https://github.com/your-username/OpenFriday.git
cd OpenFriday
```

---

### Step 2: Configure Environment Variables
Create a `.env` file in the root directory and add your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

*(Optional: Set `ANTHROPIC_API_KEY` or configure local models via Ollama if customized in `config.json`.)*

---

### Step 3: Grant Required macOS Permissions
For Open FRIDAY to track gestures and control desktop inputs, macOS requires two permissions:

1. **Camera Permission**:
   - Open **System Settings** -> **Privacy & Security** -> **Camera**.
   - Enable access for **Terminal** (or your Python / IDE executable).

2. **Accessibility Permission**:
   - Open **System Settings** -> **Privacy & Security** -> **Accessibility**.
   - Enable access for **Terminal** to allow simulated cursor navigation, clicks, and keyboard shortcuts.

---

### Step 4: Launch Open FRIDAY

Run the automated launcher script:

```bash
./run.sh
```

The launcher will automatically:
1. Verify system dependencies and Python version.
2. Initialize and configure the virtual environment (`.venv`).
3. Install or update dependencies from `requirements.txt`.
4. Verify macOS accessibility and camera permissions.
5. Launch the vision pipeline, HUD overlays, local Whisper dictation engine, and AI assistant.

---

## Configuration

You can customize runtime preferences in `config.json`:

```json
{
  "app_name": "Open FRIDAY",
  "tracking_enabled": true,
  "mouse_control_enabled": true,
  "mirror_horizontal": true,
  "camera_id": 0,
  "margin_x": 0.16,
  "margin_y": 0.16,
  "min_cutoff": 0.2,
  "beta": 0.003,
  "pinch_threshold": 0.4,
  "scroll_sensitivity": 1.0,
  "show_reticle": true,
  "show_pinch_meter": true
}
```

- `camera_id`: Webcam device index (`0` for built-in camera).
- `tracking_enabled`: Toggle computer vision gesture tracking on or off.
- `mouse_control_enabled`: Toggle system mouse injection.
- `margin_x`, `margin_y`: Edge margins for camera coordinate mapping.
- `scroll_sensitivity`: Multiplier for two-finger inertial scrolling speed.
- `show_reticle` / `show_pinch_meter`: Toggle visual HUD feedback rings.

---

## Memory Vault Structure

Open FRIDAY maintains persistent local context inside the `memory_vault/` directory:
- `user_profile.md`: Identity, user preferences, hardware configurations, and working styles.
- `projects.md`: Active code repositories, current goals, and architectural notes.
- `notes.md`: Miscellaneous facts, reminders, and reference summaries.

Files are indexed locally using semantic vector embeddings (`fastembed`). Updates to any file are automatically re-indexed in real time.

---

## Troubleshooting

- **Cursor does not move or clicks do not trigger:**
  Check **System Settings** -> **Privacy & Security** -> **Accessibility**. Ensure **Terminal** (or your terminal emulator / IDE) is enabled. If already checked, disable and re-enable it.

- **Webcam feed blank or initialization fails:**
  Ensure camera permissions are enabled in **System Settings** -> **Privacy & Security** -> **Camera**, and verify that no other background application has an exclusive lock on the camera device.

- **Voice dictation / transcription:**
  Verify that your microphone is accessible and default audio input is configured in macOS Sound Settings. On first run, `faster-whisper` downloads the model weights locally to cache.

- **Stopping an automated task immediately:**
  Make a crossed hand, wrist, or finger gesture ("X" shape) in front of the camera to instantly break any running computer automation loop.
