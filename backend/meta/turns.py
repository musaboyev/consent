"""Deduplicate actual Muse turn/final events; audio labels never become face IDs."""
from dataclasses import dataclass

@dataclass(frozen=True)
class CompletedTurn:
    turn_id: str
    text: str
    start_ms: int
    end_ms: int

class Turns:
    def __init__(self,mode="ENDPOINTING"):
        self.mode=mode
        self.current=None; self.open={}; self.seen=set(); self.fingerprints=set()

    def feed(self,event,base_ms):
        kind=event.get("type")
        offset=max(0,int(event.get("audioProcessedMs",0)))
        tid=event.get("turnId",self.current)
        if kind=="speechStart":
            self.current=event["turnId"]
            self.open[self.current]={"start":base_ms+offset}
            return None
        if kind=="speechEnd":
            self.open.setdefault(tid,{})["end"]=base_ms+offset
            return None
        # In ENDPOINTING, only speechComplete is the completed utterance.
        # A final transcript/stream summary must not consume that turn's ID.
        if kind!="speechComplete" and not (self.mode=="PUSH_TO_TALK" and kind=="transcript" and event.get("final") is True):
            return None
        text=str(event.get("transcript",""))[:2000].strip()
        if not text:return None
        key=str(tid) if tid is not None else "stream_final"
        timing=self.open.get(tid,{})
        start=timing.get("start",base_ms)
        end=timing.get("end",base_ms+offset)
        fingerprint=(text.casefold(),end)
        if key in self.seen or fingerprint in self.fingerprints:return None
        self.seen.add(key); self.fingerprints.add(fingerprint)
        return CompletedTurn(key,text,start,end)
