"""Single-owner iPhone mailbox. Reconnection never recreates the privacy engine."""
import queue
import threading
import time
import cv2
import numpy as np
from shared.contracts import FramePacket

class PhoneCamera:
    source_name='META DAT · MOCK GLASSES · LIVE IPHONE'
    def __init__(self,origin):
        self.origin=origin; self.frames=queue.Queue(maxsize=1)
        self.captured=0; self.dropped=0; self.error=None
        self.stop_event=threading.Event(); self.connected=False
        self.last_received=0.; self.connection_id=None; self.last_sequence=-1; self.last_source_ms=-1
        self.lock=threading.Lock()
        self.thread=threading.Thread(target=self.stop_event.wait,name='phone-source-owner',daemon=True)
    def start(self):self.thread.start()
    def attach(self,connection_id):
        with self.lock:
            if self.connected:raise ValueError('An iPhone source is already connected')
            self.connected=True; self.connection_id=connection_id
            self.last_sequence=-1; self.last_source_ms=-1
    def detach(self,connection_id):
        with self.lock:
            if self.connection_id==connection_id:self.connected=False
    def admit(self,jpeg,metadata,connection_id):
        with self.lock:
            if self.stop_event.is_set() or not self.connected or self.connection_id!=connection_id:
                raise ValueError('Inactive phone source')
            sequence=metadata['sequence']; source_ms=metadata['sdk_timestamp_ms']
            if type(sequence)!=int or sequence<=self.last_sequence or not isinstance(source_ms,(int,float)) or not np.isfinite(source_ms) or source_ms<self.last_source_ms:
                raise ValueError('Nonmonotonic source sequence or timestamp')
            if metadata.get('orientation')!='up' or metadata.get('provenance')!='meta_dat_mock_live_iphone':
                raise ValueError('Expected normalized Meta DAT Mock Device live-phone frames')
            image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
            if image is None or max(image.shape[:2])>1920:raise ValueError('Invalid or oversized camera image')
            # Letterbox maintains aspect ratio and stable encoder dimensions on rotation.
            h,w=image.shape[:2]; scale=min(640/w,480/h)
            resized=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
            frame=np.zeros((480,640,3),np.uint8); rh,rw=resized.shape[:2]
            y,x=(480-rh)//2,(640-rw)//2; frame[y:y+rh,x:x+rw]=resized
            now=time.monotonic(); self.captured+=1
            packet=FramePacket(self.captured,int((now-self.origin)*1000),frame,
                {'source':'meta_dat_mock_live_iphone','source_sequence':sequence,'sdk_timestamp_ms':source_ms,'connection_id':connection_id,'orientation':'up'})
            frame.setflags(write=False)
            if self.frames.full():
                try:self.frames.get_nowait(); self.dropped+=1
                except queue.Empty:pass
            self.frames.put_nowait(packet)
            self.last_sequence=sequence; self.last_source_ms=source_ms; self.last_received=now
            return packet.frame_id
    def close(self):
        self.stop_event.set(); self.thread.join(1); self.connected=False
        return not self.thread.is_alive()
