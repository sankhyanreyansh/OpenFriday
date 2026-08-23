# Open FRIDAY: Intelligent Gesture Controller and Desktop Assistant for macOS

Open FRIDAY is a hands-free desktop assistant for macOS that combines touchless webcam gesture navigation, voice dictation, conversational intelligence, and autonomous computer control.

By tracking your hands through a standard webcam and listening for voice commands, Open FRIDAY allows you to navigate macOS, trigger shortcuts, dictate text, ask visual questions about your screen, and delegate multi-step desktop tasks to an intelligent assistant.

---

## Core Capabilities

### 1. Touchless Hand Gesture Navigation
- **Precise Cursor Control**: Move your pointer naturally across the screen using your right index finger.
- **Clicking and Selection**: Perform quick pinch taps for left clicks, double pinch taps for double clicks, and held pinches for drag-and-drop operations.
- **Natural Scrolling**: Extend two fingers with your left hand raised to scroll up or down smoothly with momentum.
- **Spatial Pinch Zoom**: Pinch both hands simultaneously and stretch them apart to zoom in, or compress them closer to zoom out.
- **Workspace Navigation**: Swipe three fingers horizontally to switch between macOS Spaces, or swipe upward to open Mission Control.
- **Quick Shortcut Wheel**: Flash both palms open to bring up a radial menu with instant shortcuts for Enter, New Tab, Close Tab, and Escape.
- **Screen Snipping**: Raise your left hand and drag a box with your right pinch to snip any area of the screen for instant AI analysis.

### 2. Voice Assistant and Autonomous Desktop Automation
- **Spoken Voice Interactions**: Say "Friday, ..." followed by your question or command for hands-free voice assistance.
- **Push-to-Talk Dictation**: Hold your left hand in a fist while speaking to dictate text directly into any active search bar, text editor, or message field.
- **Smart Query Routing**: Questions are automatically routed for maximum speed and efficiency. Simple questions and math are answered immediately without taking unnecessary screenshots, while visual questions capture your screen for analysis.
- **Autonomous Desktop Automation**: Ask Open FRIDAY to complete multi-step tasks (such as opening applications, searching websites, or controlling tools) and watch the assistant navigate your interface autonomously.
- **Visual Screen Annotations**: When you ask about diagrams, complex interfaces, or code on your display, Open FRIDAY highlights and labels relevant elements directly on your screen.
- **Instant Safety Abort**: Flash both open palms at any time during automated computer control to instantly stop all actions and return control to you.

### 3. Long-Term Memory Vault
- **Persistent Personal Memory**: Open FRIDAY saves personal facts, preferences, and project notes to local text files inside the `memory_vault/` directory.
- **Context-Aware Recall**: When you ask questions relating to your profile, projects, or saved notes, Open FRIDAY searches your memory vault and incorporates relevant facts into the conversation.

---

## Gesture Reference

| Action | Hand Configuration | Description |
| :--- | :--- | :--- |
| **Move Pointer** | Right index finger extended | Move the mouse pointer across the screen. |
| **Left Click** | Right pinch tap (thumb and index finger) | Performs an immediate left click. |
| **Double Click** | Double right pinch tap within 0.4s | Performs a double click. |
| **Drag and Drop** | Right pinch hold and move | Holds down the left mouse button and drags items across the screen. |
| **Right Click** | Left hand open + Right pinch tap | Opens the context menu or performs a right click. |
| **Scroll Up / Down** | Left hand open + Right index and middle fingers extended | Move hand upward to scroll up; move downward to scroll down. |
| **Spatial Zoom** | Both hands pinched (stretch or compress) | Stretch hands apart to zoom in; compress together to zoom out. |
| **Next / Previous Space** | Right three fingers extended (swipe left/right) | Switches to the adjacent macOS desktop space. |
| **Mission Control** | Right three fingers extended (swipe up) | Opens or closes macOS Mission Control. |
| **Voice Dictation** | Left hand fist (hold while speaking) | Records your voice and types transcribed text into the active field upon release. |
| **Radial Shortcut Menu** | Both hands open palms for 0.3s | Displays on-screen wheel for Enter, New Tab, Close Tab, and Escape. |
| **Safety Abort** | Both hands open palms during automation | Instantly stops autonomous computer actions. |

---

## Local Setup Guide for macOS

### Prerequisites
- A Mac computer running macOS Monterey (12.0) or newer (Apple Silicon M1/M2/M3/M4 or Intel).
- A built-in or external USB webcam.
- Python 3.10 or Python 3.11 installed.

---

### Step 1: Clone the Repository
Open Terminal and navigate to the project directory:

```bash
cd /path/to/Open-FRIDAY
```

---

### Step 2: Configure Environment Variables
Create a `.env` file in the root directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

*Note: An OpenAI API key is required for cloud-based conversational intelligence, visual screen reasoning, and autonomous computer control.*

---

### Step 3: Grant Required macOS Permissions
For Open FRIDAY to see your gestures and move the mouse pointer, macOS requires two permissions:

1. **Camera Permission**:
   - Open **System Settings** -> **Privacy & Security** -> **Camera**.
   - Ensure that your **Terminal** app (or Python executable) is enabled.

2. **Accessibility Permission**:
   - Open **System Settings** -> **Privacy & Security** -> **Accessibility**.
   - Enable your **Terminal** app (or Python executable) to allow simulated mouse clicks and keyboard shortcuts.

---

### Step 4: Launch Open FRIDAY

Run the automated launcher script in your terminal:

```bash
./run.sh
```

The launcher script will automatically:
1. Create a local Python virtual environment (`.venv`) if one does not already exist.
2. Install all required dependencies from `requirements.txt`.
3. Check and request any missing system permissions.
4. Start the Open FRIDAY background vision engine, gesture recognizer, and interface HUD.

---

## Manual Setup (Alternative)

If you prefer to install and run the project manually without `run.sh`:

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Start the application**:
   ```bash
   python main.py
   ```

---

## Configuration and Customization

You can customize runtime settings by editing `config.json` in the root folder:

```json
{
  "app_name": "Open FRIDAY",
  "camera_id": 0,
  "tracking_enabled": true,
  "mouse_control_enabled": true,
  "scroll_sensitivity": 1.0
}
```

- `app_name`: Application branding identifier.
- `camera_id`: Selects the webcam device index (default is `0` for the built-in FaceTime camera).
- `tracking_enabled`: Enables or disables gesture tracking on startup.
- `mouse_control_enabled`: Enables or disables system mouse input dispatch.
- `scroll_sensitivity`: Adjusts sensitivity multiplier for two-finger inertial scrolling.

---

## Local Memory Vault

Open FRIDAY stores long-term memory in human-readable Markdown files inside the `memory_vault/` directory:
- `user_profile.md`: Stores personal preferences, name, and profile details.
- `projects.md`: Stores information about active projects and workflows.
- `notes.md`: Stores general facts and saved reminders.

You can inspect or edit these files in any text editor at any time.

---

## Troubleshooting

- **Cursor does not move or clicks do not register**:
  Make sure Accessibility permissions are granted to your Terminal app in **System Settings** -> **Privacy & Security** -> **Accessibility**. If the permission was previously enabled, toggle it off and back on.

- **Webcam feed is blank or crashes on launch**:
  Verify that Camera access is granted to Terminal in **System Settings** -> **Privacy & Security** -> **Camera**, and ensure no other application is exclusively locking the webcam.

- **Gesture detection accuracy**:
  Ensure your hand is clearly visible within the camera frame with adequate lighting. Avoid strong backlighting directly behind you for optimal hand landmark tracking.
