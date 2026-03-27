"""
extra_scripts.py — PlatformIO extra script
Adds a short delay after the upload step so the Nano 33 BLE has time to
re-enumerate on the USB bus before the serial monitor tries to open the port.
This prevents the [Errno 13] Permission denied race condition on Linux.
"""
import time
Import("env")  # noqa: F821 — injected by PlatformIO


def after_upload(source, target, env):  # noqa: F841
    print("Waiting 3 s for board to re-enumerate...")
    time.sleep(3)


env.AddPostAction("upload", after_upload)  # noqa: F821
