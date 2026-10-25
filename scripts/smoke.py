"""Real model binaries on a blank image; no claim of human gesture verification."""
import json
import os
from pathlib import Path
import sys
import time
import tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"consentlens-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME",str(Path(tempfile.gettempdir())/"consentlens-cache"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import mediapipe as mp
import numpy as np
import av
from vision.detectors import Detectors
from vision.engine import VisionEngine
from shared.contracts import FramePacket

def main():
    started=time.monotonic()
    engine=VisionEngine("s_model_smoke")
    image=np.zeros((480,640,3),np.uint8)
    times=[]
    for i in range(20):
        result=engine.step(FramePacket(i,i*67,image),[])
        if result.output_state=="ERROR":
            raise RuntimeError(result.reason)
        times.append(result.inference_ms)
    engine.close()
    result={"conditions":"640x480 black image, no people, real models, 20 frames; not live camera or accuracy evidence",
        "python":sys.version.split()[0],"opencv":cv2.__version__,"mediapipe":mp.__version__,"pyav":av.__version__,
        "inference_ms_p50":float(np.percentile(times,50)),"inference_ms_p95":float(np.percentile(times,95)),"elapsed_seconds":time.monotonic()-started}
    (Path(__file__).resolve().parents[1]/"evaluation"/"model-smoke.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
