"""Single mic owner in a separate thread/event loop. No network on vision worker."""
import asyncio
import copy
import json
import os
import queue
import threading
import time
import uuid
import sounddevice as sd
import numpy as np
from backend.meta.intent import Spark, safe_error, should_queue_privacy
from backend.meta.transcribe import connection
from backend.meta.budget import public_budget_status, BudgetError
from shared.contracts import PrivacyRequest
from backend.meta.turns import Turns

class Voice:
    def __init__(self,runtime):
        self.runtime=runtime; self.spark=Spark()
        self.thread=None; self.stop_event=threading.Event(); self.lock=threading.Lock()
        self.data={"status":"BLOCKED" if not os.getenv("MODEL_API_KEY") else "UNTESTED","error":"MODEL_API_KEY missing" if not os.getenv("MODEL_API_KEY") else None,
            "transcript":"","intent":None,"asr_model":os.getenv("META_ASR_MODEL","muse-voice-transcribe-1.0"),
            "spark_model":os.getenv("META_SPARK_MODEL","muse-spark-1.3"),"assumption":"Self requests assume the one visible person is speaking; no speaker-to-face identification.","real_transcripts":0}

    def update(self,**values):
        with self.lock:
            self.data.update(values)

    def status(self):
        with self.lock:
            data=copy.deepcopy(self.data)
        data["spark_status"]=self.spark.last_status
        data["spark_error"]=self.spark.last_error
        data["spark_usage"]=self.spark.last_usage
        data["budget"]=public_budget_status()
        if data["budget"]["meta_disabled"]:
            data["status"]="BUDGET_DISABLED"
            data["error"]=data["budget"]["reason"]
        return data

    def start(self):
        if not os.getenv("MODEL_API_KEY"):
            raise ValueError("Add MODEL_API_KEY to backend .env, then restart the server")
        budget_status=public_budget_status()
        if budget_status["meta_disabled"]:
            raise ValueError(budget_status["reason"])
        if self.thread and self.thread.is_alive():
            raise ValueError("Microphone already running")
        if self.runtime.state()["status"]!="RUNNING":
            raise ValueError("Start the camera session first")
        self.stop_event.clear()
        self.thread=threading.Thread(target=self._run,name="meta-voice",daemon=True)
        self.thread.start()
        return self.status()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        return self.status()

    def _run(self):
        session_id=self.runtime.session_id
        try:
            asyncio.run(self._stream(session_id))
        except Exception as exc:
            message="Meta connection timed out; check Wi-Fi and retry voice" if isinstance(exc,TimeoutError) else f"{type(exc).__name__}: {exc}"
            self.update(status="FAILED",error=safe_error(message))
        finally:
            if self.status()["status"] not in {"FAILED","BLOCKED"}:
                self.update(status="STOPPED")

    async def _stream(self,session_id):
        self.update(status="CONNECTING",error=None)
        async with connection() as (ws,provider_session):
            await self._capture_stream(session_id,ws,provider_session)

    async def _capture_stream(self,session_id,ws,provider_session):
        self.update(status="CONNECTED",provider_session_id=provider_session)
        audio=queue.Queue(maxsize=25)
        turns=Turns()
        inference=asyncio.Queue(maxsize=10)
        base_ms=self.runtime.now_ms()
        sample_count=0
        peak_max=0.; clipped_samples=0
        capture_failed=threading.Event()
        capture_error=""
        sent_samples=0
        wire_origin=None

        def callback(indata,frames,time_info,status):
            nonlocal sample_count,base_ms,capture_error,peak_max,clipped_samples
            if sample_count==0:
                # Map the first ADC sample, including the device buffer delay,
                # rather than treating stream construction as audio time zero.
                delay=max(0.,float(time_info.currentTime-time_info.inputBufferAdcTime))
                base_ms=self.runtime.now_ms()-round(delay*1000)
            if status:
                capture_error="Microphone input discontinuity: "+str(status)
                capture_failed.set()
            values=np.frombuffer(indata,dtype="<i2").astype(np.float32)
            peak=float(np.max(np.abs(values))) if len(values) else 0.
            peak_max=max(peak_max,peak)
            clipped_samples+=int(np.count_nonzero(np.abs(values)>=32767))
            self.update(microphone_peak_dbfs=round(20*np.log10(max(peak/32768,1e-6)),1),
                        microphone_peak_max_dbfs=round(20*np.log10(max(peak_max/32768,1e-6)),1),
                        microphone_clipped_samples=clipped_samples,captured_audio_ms=round((sample_count+frames)/24))
            try:
                audio.put_nowait(bytes(indata)); sample_count+=frames
            except queue.Full:
                capture_error="Outgoing microphone queue exceeded 2 seconds; network sender is behind"
                capture_failed.set()

        async def classify():
            while True:
                item=await inference.get()
                if item is None:
                    return
                turn_id,text,start,end=item
                started=time.monotonic()
                intent,route=await self.spark.classify(text,force=True)
                pid=None
                validated=self.spark.last_status=="RESPONDED"
                if not should_queue_privacy(intent,text,validated=validated):
                    self.update(intent=intent.model_dump(),intent_route=route,request_status="NO_PRIVACY_REQUEST" if validated else "PROVIDER_FAILED_NO_ACTION",target_person_id=None)
                    continue
                if intent.intent in {"PROTECT_PERSON","UNPROTECT_PERSON"} and intent.target_kind=="SELF" and not intent.needs_clarification:
                    pid=self.runtime.resolve_voice(start,end)
                if intent.intent=="UNPROTECT_PERSON" and pid is None:
                    self.update(intent=intent.model_dump(),intent_route=route,request_status="REVEAL_NEEDS_RESOLVED_TARGET",target_person_id=None)
                    continue
                request=PrivacyRequest(f"voice_{provider_session}_{turn_id}",session_id,"voice",pid,end,intent.intent if pid else "SHIELD_UNRESOLVED","VOICE_SELF_RESOLVED" if pid else "VOICE_TARGET_OR_INTENT_UNRESOLVED")
                accepted=self.runtime.enqueue(request)
                self.update(intent=intent.model_dump(),intent_route=route,request_status="QUEUED" if accepted else "STALE_SESSION",intent_processing_ms=round((time.monotonic()-started)*1000,1),target_person_id=pid)

        async def receive():
            events=0
            async for payload in ws:
                if not isinstance(payload,str):
                    continue
                event=json.loads(payload); kind=event.get("type")
                offset=int(event.get("audioProcessedMs",0))
                events+=1
                self.update(asr_processed_ms=offset,asr_events=events,asr_last_event=kind)
                if kind=="error":
                    raise RuntimeError(safe_error(event.get("message","Meta transcription failure")))
                if kind=="transcript":
                    partial=str(event.get("transcript",""))[:2000]
                    if partial.strip():
                        self.update(transcript=partial,transcript_final=False)
                completed=turns.feed(event,base_ms)
                if completed:
                    self.update(transcript=completed.text,transcript_final=True,real_transcripts=len(turns.seen),status="TRANSCRIBING")
                    try:
                        inference.put_nowait((completed.turn_id,completed.text,completed.start_ms,completed.end_ms))
                    except asyncio.QueueFull:
                        self.runtime.enqueue(PrivacyRequest("overflow_"+uuid.uuid4().hex,session_id,"voice",None,completed.end_ms,"SHIELD_UNRESOLVED"))
                        raise RuntimeError("Voice intent queue overflow; output shield requested")

        receiver=asyncio.create_task(receive()); classifier=asyncio.create_task(classify())
        try:
            with sd.RawInputStream(samplerate=24000,channels=1,dtype="int16",blocksize=1920,latency="high",callback=callback) as stream:
                self.update(status="LISTENING",microphone="ACTIVE",device_buffer_ms=round(stream.latency*1000,1))
                while not self.stop_event.is_set() and self.runtime.session_id==session_id:
                    ws.lease.check()
                    if capture_failed.is_set():
                        raise RuntimeError(capture_error+"; association timing cannot be trusted")
                    if receiver.done():
                        await receiver
                        raise RuntimeError("Meta closed the live transcription stream")
                    try:
                        chunk=audio.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(.01); continue
                    if wire_origin is None:wire_origin=time.monotonic()
                    await asyncio.sleep(max(0,wire_origin+sent_samples/24000-time.monotonic()))
                    await ws.send(chunk)
                    sent_samples+=len(chunk)//2
                    self.update(sent_audio_ms=round(sent_samples/24))
            while not audio.empty():
                await ws.send(audio.get_nowait())
            await ws.send(json.dumps({"type":"endStream"}))
            await asyncio.wait_for(receiver,15)
            await inference.put(None)
            await asyncio.wait_for(classifier,20)
        finally:
            receiver.cancel(); classifier.cancel()
            await asyncio.gather(receiver,classifier,return_exceptions=True)
            self.update(microphone="INACTIVE")
