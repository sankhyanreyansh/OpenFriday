"""
macOS Permissions and Cocoa Window Spaces Helper for FRIDAY.
Provides diagnostic checks, direct deep-links into macOS System Settings,
and native Cocoa NSWindow configuration for fullscreen spaces and window level persistence.
"""

import subprocess
import sys
from typing import Tuple
from ctypes import c_void_p


def is_accessibility_granted() -> bool:
    """
    Checks if the current process has macOS Accessibility permissions.
    Accessibility permission is required for global cursor control and synthetic clicks.
    """
    if sys.platform != 'darwin':
        return True

    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:
        try:
            cmd = "osascript -e 'tell application \"System Events\" to get name of first process'"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1.5)
            return res.returncode == 0
        except Exception:
            return False


def request_accessibility_permission() -> bool:
    """
    Triggers the macOS system prompt requesting Accessibility permissions.
    """
    if sys.platform != 'darwin':
        return True

    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
        import CoreFoundation
        options = {kAXTrustedCheckOptionPrompt: True}
        return bool(AXIsProcessTrustedWithOptions(options))
    except Exception:
        return False


def open_accessibility_settings():
    """Opens macOS System Settings -> Privacy & Security -> Accessibility."""
    try:
        subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"])
    except Exception as e:
        print(f"Error opening accessibility settings: {e}")


def open_camera_settings():
    """Opens macOS System Settings -> Privacy & Security -> Camera."""
    try:
        subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"])
    except Exception as e:
        print(f"Error opening camera settings: {e}")


def check_camera_access() -> Tuple[bool, str]:
    """
    Tests if a camera can be opened locally.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            if cap:
                cap.release()
            return False, "Could not open camera 0. Grant Camera permission in macOS System Settings."
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.size > 0:
            return True, "Camera accessible."
        else:
            return False, "Camera opened but failed to capture frame."
    except Exception as e:
        return False, f"Camera check error: {e}"


def set_macos_accessory_policy():
    """
    Sets macOS application activation policy to Accessory (Agent Mode).
    Prevents macOS from binding the application to a single desktop Space,
    making FRIDAY persistent across full-screen app transitions and Mission Control swipes.
    """
    if sys.platform != 'darwin':
        return

    try:
        import AppKit
        app = AppKit.NSApplication.sharedApplication()
        # NSApplicationActivationPolicyAccessory = 1
        app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    except Exception as e:
        print(f"[ERROR] Failed to set activation policy: {e}")


def setup_macos_fullscreen_overlay(qt_widget):
    """
    Configures underlying Cocoa NSWindow via PyObjC to float above fullscreen spaces,
    Mission Control, and all virtual desktops without auto-dismissing on focus loss.
    """
    if sys.platform != 'darwin' or qt_widget is None:
        return

    try:
        import objc
        from ctypes import c_void_p
        import Cocoa
        import Quartz

        view_ptr = int(qt_widget.winId())
        ns_view = objc.objc_object(c_void_p=view_ptr)
        ns_window = ns_view.window() if hasattr(ns_view, 'window') else None

        if ns_window:
            # 1. Elevate Window Level to ScreenSaver/PopUp level (above native full-screen shields)
            # Level 1000 (NSScreenSaverWindowLevel) or CGWindowLevelForKey(kCGScreenSaverWindowLevelKey)
            screensaver_level = Quartz.CGWindowLevelForKey(Quartz.kCGScreenSaverWindowLevelKey)
            ns_window.setLevel_(screensaver_level)

            # 2. Complete collection behavior for absolute space persistence:
            # CanJoinAllSpaces (1) | Stationary (16) | FullScreenAuxiliary (256) | IgnoresCycle (64) = 337
            behavior = (
                Cocoa.NSWindowCollectionBehaviorCanJoinAllSpaces |
                Cocoa.NSWindowCollectionBehaviorFullScreenAuxiliary |
                Cocoa.NSWindowCollectionBehaviorStationary |
                Cocoa.NSWindowCollectionBehaviorIgnoresCycle
            )
            ns_window.setCollectionBehavior_(behavior)

            # 3. Prevent deactivation & ensure orderFrontRegardless
            ns_window.setHidesOnDeactivate_(False)
            ns_window.setSharingType_(0)  # NSWindowSharingNone
            ns_window.orderFrontRegardless()
    except Exception as e:
        print(f"[ERROR] Native Cocoa overlay configuration failed: {e}")


configure_fullscreen_floating = setup_macos_fullscreen_overlay
make_window_always_float_on_top = setup_macos_fullscreen_overlay
configure_macos_spaces_persistence = setup_macos_fullscreen_overlay





