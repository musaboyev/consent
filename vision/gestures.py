"""Bilateral palm coverage + steady hold, measured in source milliseconds."""
from dataclasses import dataclass, field
import cv2
import numpy as np

PALM = [0, 1, 2, 5, 9, 13, 17]

def segment_distance(point,a,b):
    vector=b-a
    ratio=float(np.dot(point-a,vector)/max(1e-6,np.dot(vector,vector)))
    return float(np.linalg.norm(point-(a+np.clip(ratio,0,1)*vector)))

def hand_covers_eye(points,hull,eye,head_width,palm_margin=.04):
    if cv2.pointPolygonTest(hull,tuple(map(float,eye)),True)>=-head_width*palm_margin:
        return True
    # A horizontal eye-cover often uses the extended fingers rather than the
    # palm center. Include finger silhouettes, with actual segment thickness,
    # rather than filling all the gaps between spread fingertips.
    for chain in [(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]:
        for a,b in zip(chain,chain[1:]):
            if segment_distance(np.asarray(eye),points[a],points[b])<=head_width*.065:
                return True
    return False

def bilateral_cover(eyes, hands, head_width):
    if eyes is None or head_width < 20:
        return False, None
    eyes=np.asarray(eyes,dtype=np.float32)
    if eyes.shape!=(2,2) or not np.isfinite(eyes).all():
        return False,None
    separation=float(np.linalg.norm(eyes[1]-eyes[0]))
    # Collapsed pose/occluded face landmarks must not turn one eye into two.
    if not .20*head_width < separation < .75*head_width:
        return False,None
    axis=(eyes[1]-eyes[0])/separation
    regions=np.array([[eye-axis*head_width*.035,eye,eye+axis*head_width*.035] for eye in eyes])
    covered = np.zeros((2,3),dtype=bool)
    centers = []
    for hand in hands:
        points = np.asarray(hand, dtype=np.float32)
        if points.shape!=(21,2) or not np.isfinite(points).all():
            continue
        hull = cv2.convexHull(points[PALM])
        _, _, w, h = cv2.boundingRect(hull)
        if not (.22*head_width <= max(w, h) <= 2.2*head_width):
            continue
        # Closed fingertips near wrist are not the deliberate open-hand cover.
        palm_length = np.linalg.norm(points[9]-points[0])
        extended = sum(np.linalg.norm(points[tip]-points[0]) > palm_length*1.05 for tip in [8,12,16,20])
        if extended < 2 or cv2.contourArea(hull) < head_width**2*.025:
            continue
        hit = False
        for index, region in enumerate(regions):
            for sample,eye in enumerate(region):
                if hand_covers_eye(points,hull,eye,head_width,palm_margin=0):
                    covered[index,sample] = True
                    hit = True
        if hit:
            centers.append(points[PALM].mean(axis=0))
    return bool(covered.all()), np.mean(centers, axis=0) if centers else None

@dataclass
class Hold:
    hold_ms: int = 550
    accumulated: int = 0
    valid_observations: int = 0
    last_valid: int | None = None
    last_observed: int | None = None
    center: np.ndarray | None = None

    def update(self, valid, center, now, head_width):
        # A detected non-cover/one-eye frame breaks the bilateral hold. Never
        # accumulate separate one-eye events or recycled stale coverage.
        gap = now - self.last_observed if self.last_observed is not None else 0
        if not valid or center is None:
            self.accumulated=0; self.valid_observations=0; self.center=None; self.last_valid=None
            self.last_observed=now
            return 0.
        steady=self.center is None or np.linalg.norm(center-self.center)<head_width*.20
        if self.last_valid==self.last_observed and self.last_valid is not None and 0<gap<=350 and steady:
            self.accumulated+=gap
            self.valid_observations+=1
        else:
            self.accumulated=0
            self.valid_observations=1
        self.last_valid=now; self.last_observed=now; self.center=center
        # The phone may deliver 4–7 FPS even with fast local inference. Allow
        # bounded slower cadence, but require four fresh valid observations:
        # one/two sparse samples cannot masquerade as a sustained gesture.
        return min(1.,self.accumulated/self.hold_ms,self.valid_observations/4)



def peace_sign(hands, eyes, head_width):
    """V sign away from eyes: two straight fingers, ring/pinky folded."""
    for hand in hands:
        p=np.asarray(hand,dtype=np.float32)
        if p.shape!=(21,2) or not np.isfinite(p).all(): continue
        scale=float(np.linalg.norm(p[9]-p[0]))
        if not .22*head_width < scale < 1.6*head_width: continue
        def straight(mcp,pip,dip,tip):
            a=p[mcp]-p[pip]; b=p[tip]-p[pip]
            cosine=float(np.dot(a,b)/max(1e-6,np.linalg.norm(a)*np.linalg.norm(b)))
            return cosine < -.75 and np.linalg.norm(p[tip]-p[0]) > np.linalg.norm(p[pip]-p[0])*1.15
        if not (straight(5,6,7,8) and straight(9,10,11,12)): continue
        if any(np.linalg.norm(p[t]-p[0]) > np.linalg.norm(p[j]-p[0])*1.08 for j,t in [(14,16),(18,20)]): continue
        if np.linalg.norm(p[8]-p[12]) < scale*.35: continue
        if np.linalg.norm(p[4]-p[5]) > scale*.85: continue
        hull=cv2.convexHull(p[PALM])
        if eyes is None or any(hand_covers_eye(p,hull,e,head_width) for e in eyes): continue
        return True,p[9].copy()
    return False,None

@dataclass
class DeliberateAction:
    hold: Hold = field(default_factory=lambda: Hold(450))
    latched: bool = False
    released_at: int | None = None

    def update(self,valid,center,now,width):
        if not valid:
            if self.released_at is None: self.released_at=now
            if now-self.released_at>=200:
                self.latched=False
                self.hold=Hold(450)
            self.hold.update(False,None,now,width)
            return False
        self.released_at=None
        progress=self.hold.update(True,center,now,width)
        if progress>=1 and not self.latched:
            self.latched=True
            return True
        return False
