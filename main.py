#!/usr/bin/env python3
"""
FRIDAY - Minimalist macOS Hand Gesture Desktop Controller.
Dual-Hand Modifier Architecture with Fullscreen macOS Spaces Persistence,
Rock-solid Velocity-Adaptive EMA motion smoothing, and 60 FPS multi-threaded pipeline.
"""

import sys
import os
import json
import signal
import argparse

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QAction, QGuiApplication

from mouse_controller import MouseController
from vision_engine import VisionEngine
from overlay_window import TransparentOverlay
from annotation_overlay import ARAnnotationOverlay
from hud_sidebar import GlassmorphicHUDPanel, HUDContextCard
import permissions


def load_config() -> dict:
    """Loads configuration settings from config.json if available."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print(f"Loaded configuration from {config_path}")
            return cfg
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")
    return {}


def main():
    parser = argparse.ArgumentParser(description="FRIDAY - macOS Hand Gesture Desktop Controller")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index")
    parser.add_argument("--no-overlay", action="store_true", help="Disable transparent on-screen HUD reticle")
    parser.add_argument("--no-sidebar", action="store_true", help="Disable glassmorphic HUD pill")
    args = parser.parse_args()

    print("Initializing FRIDAY...")

    try:
        # Load startup configuration from config.json
        config_settings = load_config()

        # Command line arguments override config file
        cam_id = args.camera if args.camera is not None else int(config_settings.get("camera_id", 0))

        # 0. Configure macOS Accessory Activation Policy (Daemon / Agent mode)
        permissions.set_macos_accessory_policy()

        QApplication.setApplicationName("FRIDAY")
        QApplication.setOrganizationName("FRIDAY")
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        permissions.set_macos_accessory_policy()

        # Permission Diagnostics (Trigger system prompt if not granted)
        acc_ok = permissions.is_accessibility_granted()
        if not acc_ok:
            permissions.request_accessibility_permission()

        # 1. Initialize Controller & Windows
        mouse_ctrl = MouseController()
        hud_pill = GlassmorphicHUDPanel() if not args.no_sidebar else None
        hud_context_card = HUDContextCard(anchor_pill=hud_pill) if not args.no_sidebar else None
        reticle_overlay = TransparentOverlay() if not args.no_overlay else None
        annotation_overlay = ARAnnotationOverlay() if not args.no_overlay else None

        # 2. Background Vision Engine
        vision_thread = VisionEngine(camera_id=cam_id, mouse_controller=mouse_ctrl)

        # Apply startup config settings
        if config_settings:
            vision_thread.update_settings(config_settings)
            if reticle_overlay:
                reticle_overlay.update_settings(config_settings)

        # 3. Wire Qt Signal Connections
        if hud_pill:
            vision_thread.gesture_updated.connect(hud_pill.update_gesture_data)
            vision_thread.transcription_completed.connect(lambda txt: hud_pill.flash_transcribing())
            vision_thread.ai_status_changed.connect(hud_pill.update_ai_status)
            hud_pill.master_toggle_requested.connect(lambda active: setattr(vision_thread, 'tracking_enabled', active))

        if hud_context_card:
            vision_thread.context_image_captured.connect(hud_context_card.set_thumbnail)
            vision_thread.ai_response_generated.connect(hud_context_card.set_ai_reply)
            hud_context_card.context_cleared.connect(vision_thread.ai_assistant.clear_context_image)

        if reticle_overlay:
            vision_thread.gesture_updated.connect(reticle_overlay.update_gesture_data)
            reticle_overlay.dismiss_radial_requested.connect(vision_thread.dismiss_radial_menu)

        if annotation_overlay:
            vision_thread.ai_assistant.signals.annotations_ready.connect(
                lambda anns, spoken: annotation_overlay.display_annotations(anns, spoken)
            )
            vision_thread.annotations_generated.connect(annotation_overlay.display_annotations)

        # 4. macOS Menu Bar / System Tray
        tray = QSystemTrayIcon()
        tray.setIcon(app.style().standardIcon(app.style().StandardPixmap.SP_ComputerIcon))
        tray_menu = QMenu()

        toggle_tracking_action = QAction("Pause / Resume Tracking", tray)
        if hud_pill:
            toggle_tracking_action.triggered.connect(hud_pill.on_master_toggle)
        tray_menu.addAction(toggle_tracking_action)

        tray_menu.addSeparator()

        quit_action = QAction("Quit FRIDAY", tray)
        quit_action.triggered.connect(app.quit)
        tray_menu.addAction(quit_action)

        tray.setContextMenu(tray_menu)
        tray.show()

        # 5. Start Background Vision Engine
        vision_thread.start()

        # 6. Show HUD Elements
        if reticle_overlay:
            reticle_overlay.show()
            reticle_overlay.raise_()

        if hud_pill:
            hud_pill.show()
            hud_pill.raise_()

        print("FRIDAY launched successfully.")

        def cleanup():
            vision_thread.stop()
            mouse_ctrl.release_all()
            if reticle_overlay:
                reticle_overlay.close()
            if annotation_overlay:
                annotation_overlay.close()
            if hud_pill:
                hud_pill.close()
            if hud_context_card:
                hud_context_card.close()
            app.quit()

        signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
        signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())

        sys.exit(app.exec())

    except Exception as e:
        print(f"FRIDAY failed to launch: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

