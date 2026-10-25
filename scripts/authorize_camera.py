"""macOS must request AVFoundation authorization on the main thread."""
import os
import sys
import cv2

def authorize():
    print("Checking camera on main thread. Allow Camera in the macOS prompt if shown.",flush=True)
    camera=cv2.VideoCapture(int(os.getenv("CAMERA_INDEX","0")),cv2.CAP_AVFOUNDATION)
    try:
        if not camera.isOpened():
            print("Camera unavailable. Enable Camera for the launching app under System Settings > Privacy & Security > Camera, then retry.",flush=True)
            return False
        ok,frame=camera.read()
        if not ok:
            print("Camera opened but no frame arrived.",flush=True)
            return False
        print(f"Camera authorized: {frame.shape[1]}x{frame.shape[0]} frame received; preflight camera now released.",flush=True)
        return True
    finally:
        camera.release()

if __name__=="__main__":
    sys.exit(0 if authorize() else 1)
