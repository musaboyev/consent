"""Local demonstration service. Launch with one worker; browser never owns capture."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import secrets
import uuid
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")
from backend.runtime import Runtime
from backend.audio import Voice
from backend.meta.budget import public_budget_status

runtime=Runtime()
runtime.voice=Voice(runtime)
phone_token=secrets.token_urlsafe(24)

class PreviewAccessFilter(logging.Filter):
    def filter(self,record):
        return not any(isinstance(arg,str) and arg.startswith("/api/preview/protected.jpg") for arg in (record.args if isinstance(record.args,tuple) else ()))

logging.getLogger("uvicorn.access").addFilter(PreviewAccessFilter())

@asynccontextmanager
async def lifespan(app):
    recovery_path=os.getenv("CONSENTLENS_RECOVERY_SNAPSHOT")
    if recovery_path and Path(recovery_path).exists():
        recovery=json.loads(Path(recovery_path).read_text())
        if not recovery.get("session_id","").startswith("s_"):
            raise ValueError("Invalid recovery session")
        runtime.start(recovery=recovery,source=recovery.get("source","webcam"))
        consumed=Path(str(recovery_path)+".consumed")
        Path(recovery_path).rename(consumed)
        # Retain the audit of preferences without retaining biometric vectors
        # in the consumed file; the active session owns its in-memory copy.
        audit={k:v for k,v in recovery.items() if k!="identity_galleries"}
        consumed.write_text(json.dumps(audit)); consumed.chmod(0o600)
    yield
    await asyncio.to_thread(runtime.end,for_deployment=True)

app=FastAPI(title="ConsentLens local prototype",lifespan=lifespan)

def local_origin(origin):
    return not origin or urlparse(origin).hostname in {"localhost","127.0.0.1","::1","testserver"}

@app.middleware("http")
async def local_only(request: Request,call_next):
    if request.url.hostname not in {"localhost","127.0.0.1","::1","testserver"} or not local_origin(request.headers.get("origin")):
        return JSONResponse({"detail":"Local dashboard only"},status_code=403)
    response=await call_next(request)
    response.headers["Cache-Control"]="no-store"
    return response

@app.exception_handler(ValueError)
async def value_error(request,exc):
    return JSONResponse({"detail":str(exc)},status_code=409)

@app.get("/api/health")
def health():
    state=runtime.state()
    return {"service":"ConsentLens","schema_version":1,"camera":state["status"],"engine":state["status"],
        "encoder":"PyAV libx264 configured; per-pair results in recording metadata",
        "models":{name:(ROOT/"models"/name).exists() for name in ["face_detection_yunet_2023mar.onnx","face_recognition_sface_2021dec.onnx","hand_landmarker.task","pose_landmarker_lite.task"]},
        "meta":state["voice"],"error":state.get("error"),"native_meta_recording_enforcement":False}

@app.get("/api/state")
def state():
    return runtime.state()

@app.get("/api/budget")
def budget():
    return public_budget_status()

@app.post("/api/session/start")
def start_session(source: str="webcam"):
    return runtime.start(source=source)

@app.get("/api/source/iphone/pair")
def phone_pair():
    # Local-only endpoint; this upload-only token is unrelated to Model API keys.
    return {"upload_token":phone_token,"path":"/api/source/iphone/frames",
        "label":"Actual Meta DAT SDK · simulated glasses · live iPhone camera"}

@app.websocket("/api/source/iphone/frames")
async def phone_frames(websocket: WebSocket):
    # Browser origins cannot upload. Native Swift authenticates in first message,
    # never in URL/query/access logs. HTTP dashboard remains loopback-only.
    if websocket.headers.get("origin"):
        await websocket.close(code=1008); return
    await websocket.accept()
    camera=None; connection_id=uuid.uuid4().hex
    try:
        auth=await asyncio.wait_for(websocket.receive_json(),5)
        if not secrets.compare_digest(str(auth.get("upload_token","")),phone_token):
            await websocket.close(code=1008); return
        from backend.phone_source import PhoneCamera
        camera=runtime.camera
        if not isinstance(camera,PhoneCamera) or not runtime.worker or not runtime.worker.is_alive():
            await websocket.close(code=1008,reason="Start an iPhone session on the local dashboard first"); return
        camera.attach(connection_id)
        await websocket.send_json({"ready":True,"session_id":runtime.session_id})
        while True:
            metadata=await asyncio.wait_for(websocket.receive_json(),5)
            payload=await asyncio.wait_for(websocket.receive_bytes(),5)
            if len(payload)>700000:raise ValueError("JPEG exceeds 700 KB")
            frame_id=await asyncio.to_thread(camera.admit,payload,metadata,connection_id)
            await websocket.send_json({"admitted_frame_id":frame_id})
    except (WebSocketDisconnect,asyncio.TimeoutError,ValueError,KeyError,RuntimeError):
        try:await websocket.close(code=1008)
        except RuntimeError:pass
    finally:
        if camera:camera.detach(connection_id)

@app.post("/api/session/end")
def end_session():
    return runtime.end()

@app.post("/api/recording/start")
def start_recording():
    return runtime.command("recording_start")

@app.post("/api/recording/stop")
def stop_recording():
    return runtime.command("recording_stop")

@app.post("/api/source/retry")
def retry_source():
    return runtime.command("source_retry")

@app.post("/api/voice/start")
def start_voice():
    return runtime.voice.start()

@app.post("/api/voice/stop")
def stop_voice():
    return runtime.voice.stop()

@app.get("/api/preview/protected")
async def preview():
    async def frames():
        previous=-1
        while True:
            sequence,jpeg=runtime.preview()
            if jpeg and sequence!=previous:
                previous=sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg+b"\r\n"
            await asyncio.sleep(.025)
    return StreamingResponse(frames(),media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/preview/protected.jpg")
def preview_snapshot():
    sequence,jpeg=runtime.preview()
    return Response(jpeg,media_type="image/jpeg",headers={"X-Frame-Sequence":str(sequence)})

@app.websocket("/api/events")
async def events(websocket: WebSocket):
    if not local_origin(websocket.headers.get("origin")):
        await websocket.close(code=1008); return
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(runtime.state())
            await asyncio.sleep(.25)
    except (WebSocketDisconnect,RuntimeError):
        pass

@app.head("/api/recordings/{recording_id}/{kind}")
@app.get("/api/recordings/{recording_id}/{kind}")
def recording(recording_id: str,kind: str):
    if kind not in {"original","protected"}:
        raise HTTPException(404)
    entry=runtime.recordings.get(recording_id)
    if not entry or entry[1]["status"]!="FINALIZED":
        raise HTTPException(404,detail="No finalized paired recording with this ID")
    path=entry[0]/f"{kind}.mp4"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path,media_type="video/mp4",headers={"Content-Disposition":f'inline; filename="{entry[1]['session_id']}_{recording_id}_{kind}.mp4"'})

dist=ROOT/"frontend"/"dist"
if dist.exists():
    app.mount("/",StaticFiles(directory=dist,html=True),name="dashboard")
else:
    @app.get("/")
    def missing_frontend():
        return JSONResponse({"detail":"Build the dashboard: cd frontend && npm ci && npm run build. Or use npm run dev on port 5173."},status_code=503)
