"""Decode already-finalized pairs for timing/pixel QA; never modifies recordings."""
import argparse
import json
from pathlib import Path
import av
import cv2
import numpy as np

def frames(path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield float(frame.pts*frame.time_base),frame.to_ndarray(format="bgr24")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,default=Path("evaluation/artifacts"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    differences=[]; stamps=[]; mismatches=0; samples=[]
    a=iter(frames(args.directory/"original.mp4")); b=iter(frames(args.directory/"protected.mp4"))
    index=0
    while True:
        raw=next(a,None); protected=next(b,None)
        if raw is None or protected is None:
            if raw is not None or protected is not None:
                mismatches+=1
            break
        rt,ri=raw; pt,pi=protected
        if abs(rt-pt)>.001:
            mismatches+=1
        mean_difference=float(np.mean(np.abs(ri.astype(np.float32)-pi.astype(np.float32))))
        differences.append(mean_difference); stamps.append(rt)
        if index%220==0 and len(samples)<8:
            row=np.hstack([ri,pi])
            cv2.putText(row,f"Developer original | Protected   {rt:.2f}s",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
            samples.append(cv2.resize(row,(640,240)))
        index+=1
    if samples:
        cv2.imwrite(str(args.output/(args.directory.name+"-comparison.jpg")),np.vstack(samples))
    report={"recording_id":args.directory.name,"decoded_paired_frames":len(stamps),"timestamp_or_count_mismatches":mismatches,
        "first_pts_seconds":stamps[0] if stamps else None,"last_pts_seconds":stamps[-1] if stamps else None,
        "frames_with_mean_pixel_difference_above_2":sum(d>2 for d in differences),
        "mean_absolute_pixel_difference":float(np.mean(differences)) if differences else None,
        "note":"Decode/pixel evidence only; exposure, gesture intent and identity accuracy require human frame review."}
    (args.output/(args.directory.name+"-qa.json")).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
