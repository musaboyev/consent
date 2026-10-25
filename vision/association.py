import numpy as np
import cv2
from dataclasses import dataclass

def center(box):
    x,y,w,h = box
    return np.array([x+w/2, y+h/2])

def distance(a, b):
    return float(np.linalg.norm(center(a)-center(b)) / max(a[2], b[2], 20))

def iou(a, b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    intersection=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return intersection / max(1, aw*ah+bw*bh-intersection)

@dataclass
class HeadAnchor:
    box: tuple
    eyes: np.ndarray
    established_ms: int
    tracker: object = None
    tracker_box: tuple | None = None

class HeadAnchors:
    """Current-frame head anchors when eye occlusion makes pose landmarks collapse.

    Carries no person IDs or privacy state. A correlation tracker must succeed
    on every current frame; a stale rectangle alone is never accepted.
    """
    def __init__(self):
        self.previous=None; self.anchors=[]; self.previous_ms=None

    def advance(self,image,stamp,clear_faces=()):
        tracked=[]
        if self.previous is not None and self.previous_ms is not None and stamp-self.previous_ms<=250:
            for anchor in self.anchors:
                if any(distance(anchor.box,box)<.6 for box in clear_faces):
                    continue  # clear faces refresh directly; avoid tracker work
                if stamp-anchor.established_ms>8000:
                    continue
                x,y,w,h=anchor.box
                if anchor.tracker is None:
                    params=cv2.TrackerCSRT_Params()
                    params.template_size=120
                    params.number_of_scales=17
                    params.use_color_names=False
                    tracker=cv2.TrackerCSRT_create(params)
                    height,width=self.previous.shape[:2]
                    x0,y0=max(0,int(x-.1*w)),max(0,int(y-.55*h))
                    x1,y1=min(width,int(x+1.1*w)),min(height,int(y+1.05*h))
                    if x1-x0<20 or y1-y0<20:continue
                    anchor.tracker_box=(x0,y0,x1-x0,y1-y0)
                    tracker.init(self.previous,anchor.tracker_box)
                    anchor.tracker=tracker
                ok,new_box=anchor.tracker.update(image)
                if not ok:continue
                px,py,pw,ph=anchor.tracker_box
                nx,ny,nw,nh=new_box
                sx,sy=nw/pw,nh/ph
                if not .85<sx<1.18 or not .85<sy<1.18:continue
                box=(nx+(x-px)*sx,ny+(y-py)*sy,w*sx,h*sy)
                if distance(box,anchor.box)>.4:continue
                eyes=np.column_stack((nx+(anchor.eyes[:,0]-px)*sx,ny+(anchor.eyes[:,1]-py)*sy)).astype(np.float32)
                tracked.append(HeadAnchor(tuple(map(float,box)),eyes,anchor.established_ms,anchor.tracker,new_box))
        self.previous=image.copy(); self.previous_ms=stamp; self.anchors=tracked
        return tracked

    def refresh(self,observations,stamp):
        from vision.gestures import hand_covers_eye, PALM
        for obs in observations:
            if not obs.face_detected or obs.eyes is None or obs.ownership_uncertain:
                continue
            # Refresh only with uncovered, spatially plausible eyes. Hands can
            # cause YuNet false face rows; don't reset a trusted anchor to those.
            if np.linalg.norm(obs.eyes[0]-obs.eyes[1])<obs.box[2]*.20:
                continue
            if any(hand_covers_eye(h,cv2.convexHull(h[PALM]),e,obs.box[2]) for h in obs.hands for e in obs.eyes):
                continue
            nearby=[a for a in self.anchors if distance(a.box,obs.box)<.8]
            if nearby and obs.box[2]<nearby[0].box[2]*.6:
                continue
            self.anchors=[a for a in self.anchors if distance(a.box,obs.box)>=.8]
            self.anchors.append(HeadAnchor(obs.box,obs.eyes.copy(),stamp))
