"""One local runtime; HTTP readers never mutate engine state or open a camera."""
from collections import deque
from concurrent.futures import Future
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid
import cv2
import numpy as np
from backend.capture import Camera
from backend.recording import PairedRecorder
from vision.engine import VisionEngine
from vision.rendering import protect

ROOT=Path(__file__).resolve().parents[1]

class Runtime:
    def __init__(self,engine_factory=VisionEngine,camera_factory=Camera,artifact_root=None):
        self.engine_factory=engine_factory; self.camera_factory=camera_factory
        self.artifact_root=Path(artifact_root or ROOT/"recordings")
        self.lock=threading.RLock()
        self.lifecycle_lock=threading.Lock()
        self.worker=None; self.camera=None; self.engine=None
        self.session_id=None; self.origin=None
        self.stop_event=threading.Event()
        self.commands=queue.Queue(maxsize=100)
        self.requests=queue.Queue(maxsize=100)
        self.jpeg=b""; self.preview_sequence=0
        self.preview_published_at=0.
        self.snapshot={"schema_version":1,"status":"IDLE","session_id":None,"tracks":[],"events":[],"metrics":{},"recordings":[],"output_state":"SHIELDED"}
        self.recordings={}; self.recorder=None
        for metadata in self.artifact_root.glob("s_*/r_*/metadata.json"):
            try:
                summary=json.loads(metadata.read_text())
                if summary.get("recording_id")==metadata.parent.name and summary.get("session_id")==metadata.parent.parent.name:
                    self.recordings[summary["recording_id"]]=(metadata.parent,summary)
            except (OSError,ValueError,KeyError):
                continue
        self.snapshot["recordings"]=[entry[1] for entry in self.recordings.values()]
        self.history=deque(maxlen=1800)
        self.voice=None
        self.shutdown_recovery=None

    def now_ms(self):
        return int((time.monotonic()-self.origin)*1000) if self.origin else 0

    def state(self):
        with self.lock:
            state=copy.deepcopy(self.snapshot)
        state["clock_origin_monotonic"]=self.origin
        state["voice"]=self.voice.status() if self.voice else {"status":"UNAVAILABLE"}
        state["metrics"]["preview_age_ms"]=round((time.monotonic()-self.preview_published_at)*1000,1) if self.preview_published_at else None
        return state

    def start(self, recovery=None, source="webcam"):
        with self.lifecycle_lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("Session already running; reconnect uses the existing session")
            if self.camera and self.camera.thread.is_alive():
                raise ValueError("Previous camera worker has not stopped")
            self.recovery=recovery
            self.shutdown_recovery=None
            if source not in {"webcam","iphone"}:raise ValueError("Unknown camera source")
            self.source=source
            elapsed=max(0,int(recovery.get("timestamp_ms",0))) if recovery else 0
            self.origin=float(recovery["clock_origin_monotonic"]) if recovery and recovery.get("clock_origin_monotonic") is not None else time.monotonic()-elapsed/1000
            self.session_id=recovery["session_id"] if recovery else "s_"+uuid.uuid4().hex[:12]
            self.stop_event.clear(); self.history.clear()
            self.commands=queue.Queue(maxsize=100); self.requests=queue.Queue(maxsize=100)
            with self.lock:
                self.jpeg=b""; self.preview_published_at=0.
                self.snapshot={"schema_version":1,"session_id":self.session_id,"source":source,"status":"STARTING","tracks":[],"events":[],"metrics":{},"recordings":[entry[1] for entry in self.recordings.values()],"output_state":"SHIELDED","recording":None,"unresolved_privacy_request":False}
            self.worker=threading.Thread(target=self._run,name="vision-worker",daemon=True)
            self.worker.start()
            return self.state()

    def end(self,for_deployment=False):
        with self.lifecycle_lock:
            if self.voice:
                self.voice.stop()
            self.stop_event.set()
            if self.worker:
                self.worker.join(timeout=15)
                if self.worker.is_alive():
                    raise ValueError("Session still stopping; do not start another camera")
            if for_deployment and self.shutdown_recovery:
                recovery_path=os.getenv("CONSENTLENS_RECOVERY_SNAPSHOT")
                if recovery_path:
                    path=Path(recovery_path); path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
                    temporary=path.with_suffix(path.suffix+".tmp")
                    temporary.write_text(json.dumps(self.shutdown_recovery)); temporary.chmod(0o600)
                    temporary.replace(path)
            elif not for_deployment:
                self.shutdown_recovery=None
            return self.state()

    def command(self,action):
        if not self.worker or not self.worker.is_alive() or self.stop_event.is_set():
            raise ValueError("No active session")
        future=Future()
        self.commands.put_nowait((action,future))
        try:
            return future.result(timeout=15)
        except TimeoutError:
            future.cancel()
            raise ValueError("Worker did not acknowledge command; check session status") from None

    def enqueue(self,request):
        if request.session_id!=self.session_id or not self.worker or not self.worker.is_alive():
            return False
        try:
            self.requests.put_nowait(request)
            return True
        except queue.Full:
            # Safety commands cannot be silently discarded. Stop output on overload.
            self.stop_event.set()
            self.publish_slate("Privacy queue overflow")
            return False

    def resolve_voice(self,start_ms,end_ms):
        # One visible-person assumption must hold for the whole utterance interval.
        with self.lock:
            samples=[(stamp,views,state) for stamp,views,state in self.history if start_ms-100<=stamp<=end_ms+100]
        if not samples or samples[0][0]>start_ms+300 or samples[-1][0]<end_ms-300:
            return None
        ids=set()
        for stamp,views,state in samples:
            visible=[t for t in views if t["visibility"]=="VISIBLE"]
            if len(visible)!=1 or visible[0]["identity_status"]!="KNOWN" or state=="ERROR":
                return None
            ids.add(visible[0]["person_id"])
        return next(iter(ids)) if len(ids)==1 else None

    def _event(self,kind,**details):
        event={"schema_version":1,"event_id":"e_"+uuid.uuid4().hex[:12],"session_id":self.session_id,"timestamp_ms":self.now_ms(),"type":kind,"source":"runtime","person_id":None,"details":details}
        self._append_events([event])

    def _append_events(self,events):
        if not events:
            return
        with self.lock:
            self.snapshot["events"]=(self.snapshot.get("events",[])+events)[-100:]
        if self.session_id:
            directory=self.artifact_root/self.session_id
            directory.mkdir(parents=True,exist_ok=True)
            with (directory/"events.jsonl").open("a") as f:
                for event in events:
                    f.write(json.dumps(event)+"\n")

    def publish_slate(self,reason):
        image=np.zeros((480,640,3),np.uint8)
        image=protect(image,shield=True)
        ok,jpeg=cv2.imencode(".jpg",image)
        if ok:
            with self.lock:
                self.jpeg=jpeg.tobytes(); self.preview_sequence+=1
                self.preview_published_at=time.monotonic()
                self.snapshot["output_state"]="ERROR"
                self.snapshot["protection_mode"]="ERROR_SHIELD"
                self.snapshot["reason"]=reason

    def preview(self):
        with self.lock:
            if self.preview_published_at and time.monotonic()-self.preview_published_at<=1.:
                return self.preview_sequence,self.jpeg
        # A stale protected frame freezes safely, but a slate makes failure visible.
        self.publish_slate("Protected preview stale or camera unavailable")
        with self.lock:
            return self.preview_sequence,self.jpeg

    def _handle_commands(self,shape):
        while True:
            try:
                action,future=self.commands.get_nowait()
            except queue.Empty:
                return
            if not future.set_running_or_notify_cancel():
                continue
            try:
                if action=="recording_start":
                    if self.recorder:
                        raise ValueError("Already recording")
                    if shape is None:
                        raise ValueError("Wait for the first camera frame")
                    h,w=shape[:2]
                    self.recorder=PairedRecorder(self.artifact_root/self.session_id,w,h,self.session_id)
                    summary=self.recorder.summary()
                    self._event("RECORDING_STARTED",recording_id=summary["recording_id"])
                elif action=="recording_stop":
                    if not self.recorder:
                        raise ValueError("Not recording")
                    summary=self._finish_recording()
                elif action=="source_retry":
                    if self.source!="webcam":raise ValueError("Reconnect the iPhone bridge; its privacy engine is already retained")
                    if not self._restart_camera():raise ValueError("Previous camera owner is still stopping; no second camera was opened")
                    self.capture_retries=0; self.capture_retry_at=0
                    summary={"status":"RECOVERING_SOURCE","session_id":self.session_id}
                else:
                    raise ValueError("Unknown command")
                future.set_result(summary)
            except Exception as exc:
                future.set_exception(ValueError(str(exc)))

    def _restart_camera(self):
        """Worker-only restart: preserve engine/IDs/prefs and stop old owner first."""
        self.publish_slate("Camera interrupted — privacy memory retained; reconnecting capture")
        previous=self.camera
        if not previous.close():
            with self.lock:
                self.snapshot.update(status="SOURCE_ERROR",error="Previous camera owner has not stopped; refusing a second camera")
            return False
        try:
            camera=self.camera_factory(self.origin)
            camera.captured=previous.captured; camera.dropped=previous.dropped
            self.camera=camera; camera.start()
        except Exception as exc:
            self.camera=previous
            previous.error="Camera restart failed: "+type(exc).__name__
            with self.lock:self.snapshot.update(status="SOURCE_ERROR",error=previous.error)
            return False
        self.capture_wait_began=time.monotonic()
        with self.lock:self.snapshot.update(status="RECOVERING_SOURCE",error=None)
        self._event("SOURCE_RECONNECTING",preferences_retained=True)
        return True

    def _finish_recording(self):
        recorder=self.recorder
        self.recorder=None
        summary=recorder.close()
        self.recordings[recorder.recording_id]=(recorder.directory,summary)
        with self.lock:
            self.snapshot["recordings"]=[v[1] for v in self.recordings.values()]
            self.snapshot["recording"]=None
        self._event("RECORDING_STOPPED",**summary)
        return summary

    def _run(self):
        shape=None
        timings=deque(maxlen=300); ages=deque(maxlen=300); completions=deque(maxlen=300)
        processed=0; errors=0
        try:
            self.engine=self.engine_factory(self.session_id)
            if self.recovery:
                # Upgrade from an older process whose public snapshot contains no
                # embeddings. Preserve preferences; never guess face identities.
                from vision.registry import Person
                for view in self.recovery.get("tracks",[]):
                    pid=view["person_id"]
                    if not pid.startswith("p_") or not pid[2:].isdigit():continue
                    self.engine.registry.people[pid]=Person(pid,protected=view["privacy"]=="BLURRED",
                        protection_source=view.get("protection_source"),last_seen_ms=view.get("last_seen_ms",0))
                # Private, session-scoped migration data only; never API state.
                for pid,samples in self.recovery.get("identity_galleries",{}).items():
                    if pid not in self.engine.registry.people:raise ValueError("Unknown recovery identity")
                    vectors=[np.asarray(v,dtype=np.float32) for v in samples]
                    if not 1<=len(vectors)<=5 or any(v.shape!=(128,) or not np.isfinite(v).all() or abs(float(np.linalg.norm(v))-1)>.01 for v in vectors):
                        raise ValueError("Invalid recovery identity gallery")
                    self.engine.registry.people[pid].gallery=vectors
                # Old snapshots did not expose the pending latch. Missing data
                # must protect conservatively, never infer state from UI wording.
                pending=self.recovery.get("unresolved_privacy_request",True)
                if type(pending) is not bool:raise ValueError("Invalid pending privacy state")
                self.engine.unresolved=pending
                self.engine.unresolved_exemptions={pid for pid in self.recovery.get("unresolved_exemptions",[]) if pid in self.engine.registry.people}
                with self.lock:self.snapshot["unresolved_privacy_request"]=self.engine.unresolved
                self._event("SESSION_RECOVERED",preferences_preserved=True,identity_galleries_available=bool(self.recovery.get("identity_galleries")),
                    unresolved_request_preserved=self.engine.unresolved)
            if self.source=="iphone":
                from backend.phone_source import PhoneCamera
                self.camera=PhoneCamera(self.origin)
            else:self.camera=self.camera_factory(self.origin)
            if self.recovery:
                self.camera.captured=int(self.recovery.get("capture_sequence_floor",0))
                self.camera.dropped=int(self.recovery.get("dropped_before_admission",0))
            self.camera.start()
            if self.source=="iphone":
                with self.lock:self.snapshot["status"]="WAITING_SOURCE"
            self.capture_wait_began=time.monotonic()
            self.capture_retries=0; self.capture_retry_at=0
            while not self.stop_event.is_set():
                self._handle_commands(shape)
                try:
                    packet=self.camera.frames.get(timeout=.2)
                except queue.Empty:
                    if self.camera.error:
                        self.publish_slate("Camera unavailable — privacy memory retained: "+self.camera.error)
                        with self.lock:
                            self.snapshot.update(status="RECOVERING_SOURCE" if self.capture_retries<3 else "SOURCE_ERROR",error=self.camera.error)
                        if self.capture_retries<3:
                            if not self.capture_retry_at:self.capture_retry_at=time.monotonic()+min(4,2**self.capture_retries)
                            if time.monotonic()>=self.capture_retry_at:
                                self.capture_retries+=1; self.capture_retry_at=0
                                if not self._restart_camera():self.capture_retries=3
                        continue
                    if self.source=="iphone":
                        if not processed:
                            self.publish_slate("Waiting for the iPhone bridge — start it using the Mac LAN address and upload token")
                        elif not self.camera.connected:
                            self.publish_slate("iPhone source disconnected — privacy memory retained")
                        elif time.monotonic()-self.camera.last_received>1:
                            self.publish_slate("iPhone frames delayed — privacy memory retained")
                        continue
                    if time.monotonic()-self.capture_wait_began>10:
                        self.camera.error="No camera frames after 10 seconds. Check macOS Camera permission."
                    continue
                self.capture_retries=0; self.capture_retry_at=0
                self.capture_wait_began=time.monotonic()
                shape=packet.bgr.shape
                requests=[]
                while True:
                    try:
                        requests.append(self.requests.get_nowait())
                    except queue.Empty:
                        break
                result=self.engine.step(packet,requests)
                # Results too old for current-frame protection are never displayed raw.
                age=self.now_ms()-packet.timestamp_ms
                if age>1000:
                    result.protected_bgr=protect(packet.bgr,shield=True)
                    result.output_state="ERROR"; result.reason="Processing age exceeds 1000 ms"
                    result.protection_mode="ERROR_SHIELD"
                if result.output_state=="ERROR":
                    errors+=1
                if self.recorder:
                    try:
                        self.recorder.write(packet.bgr,result.protected_bgr,packet.timestamp_ms,
                            {"frame_id":packet.frame_id,"source_metadata":packet.source_metadata,"output_state":result.output_state,"reason":result.reason,"protection_mode":result.protection_mode,"unresolved_privacy_request":self.engine.unresolved,"tracks":[asdict(t) for t in result.tracks]})
                    except Exception as exc:
                        self.recorder.failed=True
                        self._event("ERROR",component="encoder",error_type=type(exc).__name__)
                        self._finish_recording()
                ok,encoded=cv2.imencode(".jpg",result.protected_bgr,[cv2.IMWRITE_JPEG_QUALITY,85])
                if not ok:
                    raise RuntimeError("Protected JPEG encoding failed")
                age=self.now_ms()-packet.timestamp_ms
                processed+=1
                timings.append(result.inference_ms); ages.append(age); completions.append(time.monotonic())
                tracks=[asdict(t) for t in result.tracks]
                with self.lock:
                    self.jpeg=encoded.tobytes(); self.preview_sequence+=1; self.preview_published_at=time.monotonic()
                    self.snapshot.update(status="RUNNING",error=None,frame_id=result.frame_id,timestamp_ms=result.timestamp_ms,
                        tracks=tracks,source=self.source,source_metadata=packet.source_metadata,output_state=result.output_state,reason=result.reason,protection_mode=result.protection_mode,unresolved_privacy_request=self.engine.unresolved,
                        recording=self.recorder.summary() if self.recorder else None,
                        metrics={"processed_frames":processed,"capture_frames":self.camera.captured,"dropped_before_admission":self.camera.dropped,
                            "vision_errors":errors,"processed_fps":round((len(completions)-1)/(completions[-1]-completions[0]),2) if len(completions)>1 else None,
                            "inference_ms_p50":round(float(np.percentile(timings,50)),1),"inference_ms_p95":round(float(np.percentile(timings,95)),1),
                            "capture_to_jpeg_age_ms_p95":round(float(np.percentile(ages,95)),1),"measurement_window_frames":len(timings)})
                    self.history.append((packet.timestamp_ms,copy.deepcopy(tracks),result.output_state))
                self._append_events([asdict(e) for e in result.events])
        except Exception as exc:
            with self.lock:
                self.snapshot["status"]="ERROR"
                self.snapshot["error"]=str(exc)
            self.publish_slate(str(exc))
            self._event("ERROR",component="runtime",message=str(exc))
        finally:
            if self.camera:
                self.camera.close()
            if self.recorder:
                self._finish_recording()
            if self.engine:
                # Snapshot authoritative flags/counters after capture has stopped.
                # Private deployment snapshot preserves this session's gallery.
                # It is never exposed through state()/HTTP or release archives.
                self.shutdown_recovery={"session_id":self.session_id,"source":self.source,
                    "timestamp_ms":self.now_ms(),"clock_origin_monotonic":self.origin,
                    "capture_sequence_floor":self.camera.captured if self.camera else 0,
                    "dropped_before_admission":self.camera.dropped if self.camera else 0,
                    "unresolved_privacy_request":self.engine.unresolved or any(t.pending_protect for t in self.engine.tracks.values()),
                    "unresolved_exemptions":sorted(self.engine.unresolved_exemptions),
                    "identity_galleries":{p.person_id:[v.tolist() for v in p.gallery] for p in self.engine.registry.people.values() if p.gallery},
                    "tracks":[{"person_id":p.person_id,"privacy":"BLURRED" if p.protected else "UNBLURRED",
                        "protection_source":p.protection_source,"last_seen_ms":p.last_seen_ms} for p in self.engine.registry.people.values()]}
                self.engine.close(); self.engine=None
            while not self.commands.empty():
                _,future=self.commands.get_nowait()
                if not future.done():
                    future.set_exception(ValueError("Session stopped"))
            with self.lock:
                if self.snapshot["status"]!="ERROR":
                    self.snapshot["status"]="IDLE"
                # Session end releases face embeddings and person preferences.
                self.snapshot["tracks"]=[]
                self.history.clear()
