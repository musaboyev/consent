"""The only camera owner. Size-one mailbox drops before admission to vision."""
import os
import queue
import threading
import time
import cv2
from shared.contracts import FramePacket

class Camera:
    def __init__(self, origin):
        self.origin=origin
        self.frames=queue.Queue(maxsize=1)
        self.stop_event=threading.Event()
        self.dropped=0; self.captured=0
        self.error=None
        self.thread=threading.Thread(target=self._run,name="camera-owner",daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        capture=None
        try:
            capture=cv2.VideoCapture(int(os.getenv("CAMERA_INDEX","0")),cv2.CAP_AVFOUNDATION)
            if not capture.isOpened():
                raise RuntimeError("Camera unavailable. Enable Camera for the terminal/Codex app in System Settings > Privacy & Security > Camera, then start a new session.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH,int(os.getenv("CAPTURE_WIDTH","640")))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT,int(os.getenv("CAPTURE_HEIGHT","480")))
            capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
            failures=0
            while not self.stop_event.is_set():
                ok,image=capture.read()
                if not ok:
                    failures+=1
                    if failures>=30:
                        raise RuntimeError("Camera stopped returning frames")
                    self.stop_event.wait(.1)
                    continue
                failures=0
                captured_at_ms=int((time.monotonic()-self.origin)*1000)
                # Normalize hardware-specific capture sizes at admission.
                width,height=int(os.getenv("CAPTURE_WIDTH","640")),int(os.getenv("CAPTURE_HEIGHT","480"))
                if image.shape[:2]!=(height,width):
                    image=cv2.resize(image,(width,height))
                self.captured+=1
                packet=FramePacket(self.captured,captured_at_ms,image)
                image.setflags(write=False)
                try:
                    self.frames.put_nowait(packet)
                except queue.Full:
                    try:
                        self.frames.get_nowait(); self.dropped+=1
                    except queue.Empty:
                        pass
                    self.frames.put_nowait(packet)
        except Exception as exc:
            self.error=str(exc)
        finally:
            if capture is not None:
                capture.release()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        return not self.thread.is_alive()
