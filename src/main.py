#!/usr/bin/env python3
"""
Open FRIDAY - macOS Hand Gesture Desktop Controller.
Dual-Hand Modifier Architecture with Fullscreen macOS Spaces Persistence,
Velocity-Adaptive Kinematic Smoothing, and 60 FPS multi-threaded vision pipeline.
"""

import sys
import signal
import argparse
import logging

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QAction

from src.control.mouse_controller import MouseController
from src.control import permissions
from src.vision.vision_engine import VisionEngine
from src.ui.overlay_window import TransparentOverlay
from src.ui.hud_sidebar import GlassmorphicStatusPill
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("OpenFriday")


def main():
    parser = argparse.ArgumentParser(description="Open FRIDAY - macOS Hand Gesture Desktop Controller")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index")
    parser.add_argument("--no-overlay", action="store_true", help="Disable transparent on-screen HUD reticle")
    parser.add_argument("--no-sidebar", action="store_true", help="Disable glassmorphic HUD pill")
    args = parser.parse_args()

    logger.info("Initializing Open FRIDAY...")

    try:
        config_settings = load_config()
        cam_id = args.camera if args.camera is not None else int(config_settings.get("camera_id", 0))

        QApplication.setApplicationName("Open FRIDAY")
        QApplication.setOrganizationName("Open FRIDAY")
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)

        # Configure macOS Accessory Activation Policy (Agent mode for Spaces persistence)
        permissions.set_macos_accessory_policy()

        # Permission Diagnostics Preflight Gate
        if not permissions.is_accessibility_granted():
            logger.warning("macOS Accessibility permission is not yet granted.")
            permissions.request_accessibility_permission()
            permissions.open_accessibility_settings()
            print("\n" + "=" * 65)
            print("[PERMISSION REQUIRED] macOS Accessibility Permission Missing")
            print("   Open FRIDAY needs Accessibility access to move cursor & click.")
            print("   Opened System Settings -> Privacy & Security -> Accessibility.")
            print("   Please grant permission and relaunch Open FRIDAY.")
            print("=" * 65 + "\n")
            sys.exit(1)

        # 1. Initialize Controller & Windows
        mouse_ctrl = MouseController()
        hud_pill = GlassmorphicStatusPill() if not args.no_sidebar else None
        reticle_overlay = TransparentOverlay() if not args.no_overlay else None

        # 2. Background Vision Engine
        vision_thread = VisionEngine(
            camera_id=cam_id,
            mouse_controller=mouse_ctrl,
        )

        if config_settings:
            vision_thread.update_settings(config_settings)
            if reticle_overlay:
                reticle_overlay.update_settings(config_settings)

        # 3. Wire Qt Signal Connections
        if hud_pill:
            vision_thread.gesture_updated.connect(hud_pill.update_gesture_data)
            vision_thread.status_changed.connect(hud_pill.on_status_changed)
            hud_pill.master_toggle_requested.connect(lambda active: setattr(vision_thread, 'tracking_enabled', active))

        if reticle_overlay:
            vision_thread.gesture_updated.connect(reticle_overlay.update_gesture_data)

        # 4. macOS Menu Bar / System Tray
        tray = QSystemTrayIcon()
        tray.setIcon(app.style().standardIcon(app.style().StandardPixmap.SP_ComputerIcon))
        tray_menu = QMenu()

        toggle_tracking_action = QAction("Pause / Resume Tracking", tray)
        if hud_pill:
            toggle_tracking_action.triggered.connect(hud_pill.on_master_toggle)
        tray_menu.addAction(toggle_tracking_action)

        tray_menu.addSeparator()

        quit_action = QAction("Quit Open FRIDAY", tray)
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

        logger.info("Open FRIDAY launched successfully.")

        def cleanup():
            logger.info("Shutting down Open FRIDAY...")
            vision_thread.stop()
            mouse_ctrl.release_all()
            if reticle_overlay:
                reticle_overlay.close()
            if hud_pill:
                hud_pill.close()
            app.quit()

        signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
        signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())

        sys.exit(app.exec())

    except Exception as e:
        logger.exception("Open FRIDAY failed to launch: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
