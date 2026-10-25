"""Bounded REAL microphone → Meta diagnostic, no engine injection or recording."""
import asyncio
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
from backend.meta.transcribe import connection
from backend.meta.intent import safe_error
from backend.meta.budget import public_budget_status

async def main():
    load_dotenv(Path(__file__).resolve().parents[1]/".env")
    report={"source":"Real microphone / Meta diagnostic; not an integrated privacy activation","events":[],"blocks":0,"overflows":0,"peak_dbfs":-120.,"transcripts":[]}
    try:
        async with connection(max_seconds=20) as (ws,sid):
            async def receive():
                async for payload in ws:
                    if not isinstance(payload,str):continue
                    event=json.loads(payload)
                    report["events"].append({k:event[k] for k in ["type","audioProcessedMs","turnId","final"] if k in event})
                    if event.get("type")=="error":raise RuntimeError(safe_error(event.get("message","Meta error")))
                    if event.get("transcript"):
                        text=safe_error(event["transcript"])
                        if not report["transcripts"] or text!=report["transcripts"][-1]:
                            report["transcripts"].append(text)
                            print("REAL Muse transcript:",text,flush=True)
            receiver=asyncio.create_task(receive())
            try:
                with sd.RawInputStream(samplerate=24000,channels=1,dtype="int16",blocksize=1920,latency="high") as stream:
                    report["device_buffer_ms"]=stream.latency*1000
                    print("Microphone diagnostic active for 12 seconds. Speak a privacy request now.",flush=True)
                    started=time.monotonic(); sent=0
                    while time.monotonic()-started<12:
                        data,overflow=await asyncio.to_thread(stream.read,1920)
                        if overflow:report["overflows"]+=1
                        chunk=bytes(data)
                        values=np.frombuffer(chunk,dtype="<i2").astype(np.float32)
                        db=20*np.log10(max(float(np.max(np.abs(values)))/32768,1e-6))
                        report["peak_dbfs"]=max(report["peak_dbfs"],float(db))
                        report["blocks"]+=1
                        # CoreAudio may deliver buffered blocks in bursts. Pace the wire.
                        await asyncio.sleep(max(0,started+sent/24000-time.monotonic()))
                        await ws.send(chunk); sent+=len(chunk)//2
                        if receiver.done():await receiver
                await ws.send(json.dumps({"type":"endStream"}))
                await asyncio.wait_for(receiver,15)
                report["status"]="COMPLETED"
            finally:
                receiver.cancel(); await asyncio.gather(receiver,return_exceptions=True)
    except Exception as exc:
        report["status"]="FAILED"; report["error"]=safe_error(f"{type(exc).__name__}: {exc}")
    report["budget"]=public_budget_status()
    output=Path(__file__).resolve().parents[1]/"evaluation/microphone-diagnostic.json"
    output.write_text(json.dumps(report,indent=2)+"\n")
    summary={k:v for k,v in report.items() if k!="events"}
    summary["event_count"]=len(report["events"])
    summary["server_processed_ms"]=max((e.get("audioProcessedMs",0) for e in report["events"]),default=0)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":asyncio.run(main())
