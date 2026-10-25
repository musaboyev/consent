"""Two live encoders, one source timestamp sequence. Stop only flushes."""
from fractions import Fraction
import json
from pathlib import Path
import uuid
import av

class PairedRecorder:
    def __init__(self, directory, width, height, session_id):
        self.recording_id = "r_"+uuid.uuid4().hex[:12]
        self.directory=Path(directory)/self.recording_id
        self.directory.mkdir(parents=True,exist_ok=False)
        self.session_id=session_id
        self.containers=[]; self.streams=[]
        self.frames=0; self.first_ms=None; self.last_ms=None
        self.failed=False; self.closed=False
        self.width=width; self.height=height
        try:
            for kind in ["original","protected"]:
                container=av.open(str(self.directory/f"{kind}.mp4"),mode="w",options={"movflags":"+faststart"})
                self.containers.append(container)
                stream=container.add_stream("libx264",rate=30)
                stream.width=width; stream.height=height; stream.pix_fmt="yuv420p"
                stream.time_base=Fraction(1,1000)
                stream.codec_context.time_base=Fraction(1,1000)
                stream.options={"preset":"ultrafast","crf":"19","tune":"zerolatency"}
                self.streams.append(stream)
        except Exception:
            self.failed=True
            self.close()
            raise
        self._metadata()
        self.journal=(self.directory/"frames.jsonl").open("a")

    def write(self, raw, protected, timestamp_ms, decision=None):
        if self.closed or self.failed:
            raise RuntimeError("Recording is unavailable")
        if raw.shape != protected.shape or raw.shape[:2] != (self.height,self.width):
            self.failed=True
            raise RuntimeError("Recording dimensions changed")
        if self.first_ms is None:
            self.first_ms=timestamp_ms
        # Require strictly increasing input timestamps, never flatten variable timing.
        if self.last_ms is not None and timestamp_ms<=self.last_ms:
            raise RuntimeError("Nonmonotonic recording timestamp")
        try:
            for image,container,stream in zip([raw,protected],self.containers,self.streams):
                frame=av.VideoFrame.from_ndarray(image,format="bgr24")
                frame.pts=timestamp_ms-self.first_ms
                frame.time_base=Fraction(1,1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            self.frames+=1; self.last_ms=timestamp_ms
            self.journal.write(json.dumps({"frame_index":self.frames-1,"timestamp_ms":timestamp_ms,"pts_ms":timestamp_ms-self.first_ms,**(decision or {})})+"\n")
        except Exception:
            self.failed=True
            raise

    def close(self):
        if not self.closed:
            for container,stream in zip(self.containers,self.streams):
                try:
                    for packet in stream.encode():
                        container.mux(packet)
                except Exception:
                    self.failed=True
                finally:
                    try:
                        container.close()
                    except Exception:
                        self.failed=True
            self.closed=True
            if hasattr(self,"journal"):
                self.journal.close()
            self._metadata()
        return self.summary()

    def summary(self):
        return {"recording_id":self.recording_id,"session_id":self.session_id,"frames":self.frames,
                "first_timestamp_ms":self.first_ms,"last_timestamp_ms":self.last_ms,
                "duration_ms":(self.last_ms-self.first_ms) if self.last_ms is not None else 0,
                "status":"FAILED" if self.failed or (self.closed and self.frames==0) else "FINALIZED" if self.closed else "RECORDING",
                "audio":False}

    def _metadata(self):
        (self.directory/"metadata.json").write_text(json.dumps(self.summary(),indent=2)+"\n")
