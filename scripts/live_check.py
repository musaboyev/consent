"""Real local-service observation. No privacy-event or transcript injection."""
import argparse
import json
from pathlib import Path
import time
import httpx

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seconds",type=int,default=60)
    parser.add_argument("--require-source",choices=["iphone","webcam"])
    parser.add_argument("--fresh-activation",action="store_true",help="Refuse an already-protected/uncertain session; never clear it")
    parser.add_argument("--require-return",action="store_true")
    args=parser.parse_args()
    if not 1<=args.seconds<=600:raise SystemExit("Use 1–600 seconds")
    root=Path(__file__).resolve().parents[1]
    states=[]; observed=set()
    with httpx.Client(base_url="http://127.0.0.1:8000",timeout=20,trust_env=False) as client:
        state=client.get("/api/state").json()
        if state["status"] in {"IDLE","ERROR"}:
            response=client.post("/api/session/start",params={"source":args.require_source or "webcam"}); response.raise_for_status()
        deadline=time.monotonic()+20
        while state["status"]!="RUNNING":
            if time.monotonic()>deadline:raise SystemExit("Camera pipeline did not become RUNNING")
            time.sleep(.2); state=client.get("/api/state").json()
            if state["status"]=="ERROR":raise SystemExit(state.get("error","Camera error"))
        if state.get("recording"):raise SystemExit("Already recording; existing recording left running")
        if args.require_source and state.get("source")!=args.require_source:
            raise SystemExit("Wrong active source; existing session and privacy preferences left unchanged")
        if args.fresh_activation and (state.get("output_state")!="NORMAL" or any(t["privacy"]=="BLURRED" for t in state["tracks"])):
            raise SystemExit("Fresh-activation check requires an intentionally fresh, resolved session; no privacy state was cleared")
        observed={event["event_id"] for event in state.get("events",[])}
        new_events=[]
        response=client.post("/api/recording/start"); response.raise_for_status()
        recording=response.json()
        print("Live paired recording started:",recording["recording_id"],flush=True)
        try:
            deadline=time.monotonic()+args.seconds
            while time.monotonic()<deadline:
                state=client.get("/api/state").json()
                states.append({k:state.get(k) for k in ["session_id","source","source_metadata","status","frame_id","timestamp_ms","tracks","metrics","output_state","protection_mode","unresolved_privacy_request","reason","error","recording"]})
                for event in state.get("events",[]):
                    if event["event_id"] not in observed:
                        observed.add(event["event_id"])
                        new_events.append(event)
                        if event["type"] in {"PROTECTION_ACTIVATED","PERSON_REIDENTIFIED","ERROR"}:
                            print(event["type"],event.get("person_id"),event["timestamp_ms"],flush=True)
                if state["status"]=="ERROR":break
                time.sleep(.5)
        finally:
            current=client.get("/api/state").json()
            if current.get("recording",{} ) and current["recording"]["recording_id"]==recording["recording_id"]:
                stopped=client.post("/api/recording/stop")
                finalized=stopped.json()
            else:
                finalized=next((r for r in current.get("recordings",[]) if r["recording_id"]==recording["recording_id"]),{"recording_id":recording["recording_id"],"status":"NOT_FOUND"})
            final=client.get("/api/state").json()
            activated={e["person_id"] for e in new_events if e["type"]=="PROTECTION_ACTIVATED" and e.get("source")=="gesture"}
            restored={e["person_id"] for e in new_events if e["type"]=="PERSON_REIDENTIFIED" and e.get("details",{}).get("restored_privacy")=="BLURRED"}
            gates={"source_matches":not args.require_source or all(s.get("source")==args.require_source for s in states),
                "new_gesture_activation":bool(activated),"same_protected_id_returned":bool(activated & restored),
                "pair_finalized":finalized.get("status")=="FINALIZED"}
            report={"source":"LIVE "+str(final.get("source","unknown"))+" through running ConsentLens service; no injected requests or transcripts",
                "recording":finalized,"states":states,"events":new_events,"observed_engine_event_checks":gates,
                "manual_checks_still_required":["Physical iPhone SDK callback provenance","Intended gesture physically performed","Pixel review for face exposure and neighbor treatment","Decode both videos and compare PTS"],
                "budget":final.get("voice",{}).get("budget")}
            output=root/"evaluation"/("live-"+recording["recording_id"]+".json")
            output.write_text(json.dumps(report,indent=2)+"\n")
            print("Finalization:",json.dumps(finalized),flush=True)
            print("Observation log:",output.name,flush=True)
            print("Observed engine events (not a substitute for human pixel review):",json.dumps(gates),flush=True)
            failed=not gates["source_matches"] or not gates["pair_finalized"] or (args.fresh_activation and not gates["new_gesture_activation"]) or (args.require_return and not gates["same_protected_id_returned"])
        if failed:raise SystemExit("Required observation was not demonstrated; report retained")

if __name__=="__main__":main()
