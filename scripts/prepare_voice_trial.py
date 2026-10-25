"""Start a camera/real microphone trial without injecting any privacy request."""
import time
import httpx

with httpx.Client(base_url="http://127.0.0.1:8000",timeout=20,trust_env=False) as client:
    state=client.get("/api/state").json()
    if state["status"] in {"IDLE","ERROR"}:
        client.post("/api/session/start").raise_for_status()
    deadline=time.monotonic()+20
    while True:
        state=client.get("/api/state").json()
        if state["status"]=="RUNNING":break
        if state["status"]=="ERROR" or time.monotonic()>deadline:raise SystemExit(state.get("error","Camera did not start"))
        time.sleep(.2)
    if any(t["privacy"]=="BLURRED" for t in state["tracks"]):
        raise SystemExit("Existing protection retained. End the development session explicitly before a fresh voice-only activation trial.")
    client.post("/api/voice/start").raise_for_status()
    if not state.get("recording"):
        client.post("/api/recording/start").raise_for_status()
    deadline=time.monotonic()+25
    while True:
        state=client.get("/api/state").json()
        voice=state["voice"]
        if voice["status"] in {"LISTENING","TRANSCRIBING"}:
            print("READY: real Muse microphone is listening. Keep hands down and say Blur me.",flush=True)
            print("Session:",state["session_id"],"Recording:",state["recording"]["recording_id"],flush=True)
            break
        if voice["status"] in {"FAILED","BUDGET_DISABLED"} or time.monotonic()>deadline:
            raise SystemExit(voice.get("error") or "Voice did not become ready")
        time.sleep(.2)
