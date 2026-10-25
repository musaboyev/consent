"""Read-only offline gesture timing; never a fresh live trial or video blur pass."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av,numpy as np
from vision.engine import VisionEngine
from vision.gestures import bilateral_cover,peace_sign
from shared.contracts import FramePacket
parser=argparse.ArgumentParser()
parser.add_argument('clip',type=Path)
parser.add_argument('--start',type=float,default=108)
parser.add_argument('--end',type=float,default=132)
parser.add_argument('--output',type=Path,default=Path('evaluation/gesture-timing-recheck.json'))
args=parser.parse_args();clip=args.clip
engine=VisionEngine('s_OFFLINE_TIMING');original_detect=engine.detectors.detect;rows=[];current={}
def detect(image,stamp):
 obs=original_detect(image,stamp)
 current['cover']=any(bilateral_cover(o.eyes,o.hands,o.box[2])[0] and not o.ownership_uncertain for o in obs)
 current['peace']=any(peace_sign(o.hands,o.eyes,o.box[2])[0] and not o.ownership_uncertain and o.quality for o in obs)
 return obs
engine.detectors.detect=detect
start=time.monotonic()
with av.open(str(clip)) as c:
 for i,f in enumerate(c.decode(video=0)):
  stamp=round(float(f.pts*f.time_base)*1000)
  if stamp<args.start*1000:continue
  if stamp>args.end*1000:break
  stamp-=round(args.start*1000)
  r=engine.step(FramePacket(i,stamp,f.to_ndarray(format='bgr24')),[])
  rows.append({'ms':stamp,**current,'progress':max([t.progress for t in engine.tracks.values()],default=0),'events':[e.type for e in r.events if e.type in ['PROTECTION_ACTIVATED','PROTECTION_REMOVED']]})
engine.close()
events=[r for r in rows if r['events']];report={'evidence':'OFFLINE real detector replay of saved iPhone original, not a fresh live trial','clip':str(clip),'source_interval_seconds':[args.start,args.end],'frames':len(rows),'compute_seconds':round(time.monotonic()-start,3),'eye_hold_ms':engine.hold_ms,'first_cover_ms':next((r['ms'] for r in rows if r['cover']),None),'first_peace_ms':next((r['ms'] for r in rows if r['peace']),None),'events':events,'rows':rows}
args.output.parent.mkdir(exist_ok=True,parents=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
