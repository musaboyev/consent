"""Current hand landmarks with bounded continuity through pose-wrist dropouts."""
import numpy as np
from vision.association import distance

class HandOwnership:
    def __init__(self):
        self.direct=[]

    def assign(self,hands,observations,now):
        # Borrowed assignments never extend the direct-body evidence lifetime.
        self.direct=[r for r in self.direct if 0<=now-r[0]<=250]
        fresh=[];unassigned=[]
        for points in hands:
            owners=[]
            for i,obs in enumerate(observations):
                if obs.wrists is not None and len(obs.wrists):
                    d=min(np.linalg.norm(points[0]-w) for w in obs.wrists)/max(obs.box[2],20)
                    if d<.8:owners.append((float(d),i))
            owners.sort()
            owner=None
            if owners and (len(owners)==1 or owners[1][0]-owners[0][0]>.22):
                owner=owners[0][1]
                fresh.append((now,points[0].copy(),observations[owner].box))
            elif not owners:
                candidates=set()
                for stamp,wrist,box in self.direct:
                    if np.linalg.norm(points[0]-wrist)>max(box[2],20)*.35:continue
                    matches=sorted((distance(o.box,box),i) for i,o in enumerate(observations))
                    if not matches or matches[0][0]>.35:continue
                    if len(matches)>1 and matches[1][0]-matches[0][0]<.25:continue
                    candidates.add(matches[0][1])
                if len(candidates)==1:owner=next(iter(candidates))
            if owner is None:unassigned.append(points)
            else:observations[owner].hands.append(points)
        self.direct=(self.direct+fresh)[-64:]
        return unassigned
