#!/usr/bin/env python3
"""
FRIDAY - Minimalist macOS Hand Gesture Desktop Controller.
Dual-Hand Modifier Architecture with Fullscreen macOS Spaces Persistence,
Rock-solid Velocity-Adaptive EMA motion smoothing, and 60 FPS multi-threaded pipeline.
"""

import sys
import os
import signal
import argparse

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QAction, QGuiApplication

from mouse_controller import MouseController
from vision_engine import VisionEngine
from overlay_window import TransparentOverlay
from hud_sidebar import GlassmorphicHUDPanel
import permissions


def main():
    parser = argparse.ArgumentParser(description="FRIDAY - macOS Hand Gesture Desktop Controller")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--no-overlay", action="store_true", help="Disable transparent on-screen HUD reticle")
    parser.add_argument("--no-sidebar", action="store_true", help="Disable glassmorphic HUD pill")
    args = parser.parse_args()

    # 0. Configure macOS Accessory Activation Policy (Daemon / Agent mode)
    permissions.set_macos_accessory_policy()

    QApplication.setApplicationName("FRIDAY")
    QApplication.setOrganizationName("FRIDAY")
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    permissions.set_macos_accessory_policy()

    print("=" * 65)
    print("🤖 Initializing FRIDAY (Minimalist Glassmorphic Controller)...")
    print("=" * 65)

    # 1. Screen Detection
    primary_screen = QGuiApplication.primaryScreen()
    screen_geo = primary_screen.geometry()
    print(f"🖥️  Display Detected: {screen_geo.width()}x{screen_geo.height()} (DPI Scale: {primary_screen.devicePixelRatio():.1f}x)")

    # 2. Permission Diagnostics
    acc_ok = permissions.is_accessibility_granted()
    if not acc_ok:
        print("⚠️  [NOTICE] macOS Accessibility permission is not yet granted.")
        print("    FRIDAY requires Accessibility permissions to move the global mouse cursor.")
        print("    Triggering system prompt now...")
        permissions.request_accessibility_permission()
    else:
        print("✓ macOS Accessibility: Granted")

    cam_ok, cam_msg = permissions.check_camera_access()
    if not cam_ok:
        print(f"⚠️  [NOTICE] Camera check: {cam_msg}")
    else:
        print(f"✓ Camera Access: {cam_msg}")

    # 3. Initialize Controller & Windows
    mouse_ctrl = MouseController()
    hud_pill = GlassmorphicHUDPanel() if not args.no_sidebar else None
    reticle_overlay = TransparentOverlay() if not args.no_overlay else None

    # Background Vision Engine with hardcoded optimal constants
    vision_thread = VisionEngine(camera_id=args.camera, mouse_controller=mouse_ctrl)

    # 4. Wire Qt Signal Connections
    if hud_pill:
        vision_thread.gesture_updated.connect(hud_pill.update_gesture_data)
        hud_pill.master_toggle_requested.connect(lambda active: setattr(vision_thread, 'tracking_enabled', active))

    if reticle_overlay:
        vision_thread.gesture_updated.connect(reticle_overlay.update_gesture_data)

    # 5. macOS Menu Bar / System Tray
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

    # 6. Start Background Vision Engine
    vision_thread.start()

    # 7. Show HUD Elements
    if reticle_overlay:
        reticle_overlay.show()
        reticle_overlay.raise_()

    if hud_pill:
        hud_pill.show()
        hud_pill.raise_()

    print("=" * 65)
    print("✓ FRIDAY is fully operational and visible on screen!")
    if hud_pill:
        print(f"  - Glassmorphic Status Pill: Visible={hud_pill.isVisible()}, Geo={hud_pill.geometry().width()}x{hud_pill.geometry().height()} at ({hud_pill.geometry().x()}, {hud_pill.geometry().y()})")
    if reticle_overlay:
        print(f"  - Click-Through Reticle Overlay: Visible={reticle_overlay.isVisible()}, Geo={reticle_overlay.geometry().width()}x{reticle_overlay.geometry().height()}")
    print("  - Gestures: Right Pinch = Left Click | Pinch & Hold = Drag | Left Hand Up + Right Pinch = Right Click | Left Hand Up + 2 Fingers = Scroll | 3 Fingers Swipe = Spaces / Mission Control")
    print("  - Press Ctrl+C in terminal or Quit in Menu Bar to exit.")
    print("=" * 65)

    def cleanup():
        print("\nStopping FRIDAY...")
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


if __name__ == "__main__":
    main()
