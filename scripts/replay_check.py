"""Offline regression using real detectors, never a live demonstration substitute."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av
from shared.contracts import FramePacket
from vision.engine import VisionEngine

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("video",type=Path)
    parser.add_argument("--end",type=float,default=30)
    parser.add_argument("--output",type=Path,default=Path("evaluation/artifacts/replay-check.json"))
    args=parser.parse_args()
    engine=VisionEngine("s_OFFLINE_REGRESSION")
    events=[]; count=0; protected_frames=0; shielded=0; started=time.monotonic()
    with av.open(str(args.video)) as container:
        for frame in container.decode(video=0):
            stamp=int(float(frame.pts*frame.time_base)*1000)
            if stamp>args.end*1000:break
            result=engine.step(FramePacket(count,stamp,frame.to_ndarray(format="bgr24")),[])
            events.extend(asdict(e) for e in result.events)
            count+=1
            protected_frames+=any(p.protected for p in engine.registry.people.values())
            shielded+=result.output_state!="NORMAL"
    report={"source":"OFFLINE REGRESSION — real detectors on recorded original; no protected video produced",
        "input":str(args.video),"frames":count,"elapsed_seconds":time.monotonic()-started,
        "frames_after_any_activation":protected_frames,"shielded_frames":shielded,"events":events}
    engine.close()
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="events"},indent=2))
    print("Activation events:",[e for e in events if e["type"]=="PROTECTION_ACTIVATED"])

if __name__=="__main__":main()
