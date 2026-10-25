"""Explicit offline detector diagnostic. Does not create protected recordings."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av
import cv2
import numpy as np
from vision.detectors import Detectors
from vision.gestures import bilateral_cover, PALM

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("video",type=Path)
    parser.add_argument("--start",type=float,default=5)
    parser.add_argument("--end",type=float,default=25)
    parser.add_argument("--step-ms",type=int,default=150)
    args=parser.parse_args()
    detector=Detectors(); diagnostics=[]; samples=[]; last=-1000
    output=Path("evaluation/artifacts"); output.mkdir(exist_ok=True,parents=True)
    with av.open(str(args.video)) as container:
        for frame in container.decode(video=0):
            seconds=float(frame.pts*frame.time_base)
            if seconds<args.start: continue
            if seconds>args.end: break
            stamp=int(seconds*1000)
            if stamp-last<args.step_ms: continue
            last=stamp
            image=frame.to_ndarray(format="bgr24")
            observations=detector.detect(image,stamp)
            rows=[]
            for obs in observations:
                valid,_=bilateral_cover(obs.eyes,obs.hands,obs.box[2])
                all_valid,_=bilateral_cover(obs.eyes,detector.last_hands,obs.box[2])
                distances=[]
                for hand in detector.last_hands:
                    hull=cv2.convexHull(hand[PALM])
                    distances.append([round(cv2.pointPolygonTest(hull,tuple(map(float,e)),True),1) for e in obs.eyes] if obs.eyes is not None else [])
                    cv2.polylines(image,[hull.astype(np.int32)],True,(0,255,255),2)
                    for point in hand:
                        cv2.circle(image,tuple(point.astype(int)),2,(0,255,255),-1)
                if obs.eyes is not None:
                    for eye in obs.eyes:
                        cv2.circle(image,tuple(eye.astype(int)),5,(0,0,255),2)
                rows.append({"head_width":round(obs.box[2],1),"eye_span":float(np.linalg.norm(obs.eyes[0]-obs.eyes[1])) if obs.eyes is not None else None,"face_detected":obs.face_detected,"quality":obs.quality,"hands_owned":len(obs.hands),"ownership_uncertain":obs.ownership_uncertain,"owned_cover":valid,"any_cover":all_valid,"palm_distances_to_eyes":distances})
            diagnostics.append({"seconds":round(seconds,3),"flow_heads":detector.last_flow_heads,"hands":len(detector.last_hands),"pose_wrists":detector.last_pose_wrists,"observations":rows})
            if len(samples)<10 and (not samples or seconds-args.start>=len(samples)*2):
                label=f"OFFLINE DIAGNOSTIC {seconds:.2f}s hands {len(detector.last_hands)}"
                cv2.putText(image,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                samples.append(cv2.resize(image,(480,360)))
    detector.close()
    (output/"gesture-diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    if samples: cv2.imwrite(str(output/"gesture-diagnostics.jpg"),np.vstack(samples))
    print(json.dumps({"sampled_frames":len(diagnostics),"frames_with_any_cover":sum(any(o["any_cover"] for o in d["observations"]) for d in diagnostics),"frames_with_owned_cover":sum(any(o["owned_cover"] for o in d["observations"]) for d in diagnostics),"frames_with_hands":sum(d["hands"]>0 for d in diagnostics)},indent=2))

if __name__=="__main__":main()
