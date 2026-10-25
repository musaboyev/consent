"""Single state owner. Motion tracks expire; privacy and face galleries do not."""
from dataclasses import dataclass, field
import os
import time
import uuid
import numpy as np
from shared.contracts import FrameResult, PrivacyEvent, TrackView
from vision.association import distance, iou
from vision.gestures import Hold, bilateral_cover, peace_sign, DeliberateAction
from vision.identity import select, similarity, rank
from vision.registry import Registry
from vision.rendering import protect, protect_scene, head_box

@dataclass
class Track:
    track_id: str
    box: tuple
    last_seen: int
    person_id: str | None = None
    match_id: str | None = None
    agreements: int = 0
    observations: int = 0
    hold: Hold = field(default_factory=Hold)
    reveal: DeliberateAction = field(default_factory=DeliberateAction)
    pending_protect: bool = False
    owner_uncertain_confirmed: bool = False
    identity_uncertain: bool = False
    progress: float = 0.
    enrollment_samples: list = field(default_factory=list)
    enrollment_last_ms: int = -1000
    reliable_box: tuple | None = None
    reliable_at: int = -1000

class VisionEngine:
    def __init__(self, session_id, detectors=None):
        if detectors is None:
            from vision.detectors import Detectors
            detectors = Detectors()
        self.detectors = detectors
        self.threshold = float(os.getenv("IDENTITY_THRESHOLD", ".45"))
        self.margin = float(os.getenv("IDENTITY_MARGIN", ".08"))
        self.hold_ms = int(os.getenv("GESTURE_HOLD_MS", "550"))
        self.reset(session_id)

    def reset(self, session_id):
        self.session_id = session_id
        self.registry = Registry()
        self.tracks: dict[str, Track] = {}
        self.unresolved = False
        self.unresolved_exemptions=set()
        self.seen_requests = set()
        self.output_state = "NORMAL"
        self.protection_mode = "TARGETED"
        self.events = []
        self.temporary_tracks=set()

    def event(self, kind, now, pid=None, source="vision", **details):
        self.events.append(PrivacyEvent("e_"+uuid.uuid4().hex[:12],self.session_id,kind,now,pid,source,details))

    def mark_unresolved(self):
        self.unresolved=True
        self.unresolved_exemptions.clear()

    def activate(self, pid, source, now):
        if self.registry.protect(pid,source):
            self.event("PROTECTION_ACTIVATED",now,pid,source)

    def deactivate(self, pid, source, now):
        if self.unresolved:
            self.unresolved_exemptions.add(pid)
            self.event("PENDING_REQUEST_RELEASED_FOR_PERSON",now,pid,source)
        if self.registry.unprotect(pid,source):
            self.event("PROTECTION_REMOVED",now,pid,source)

    def step(self, packet, requests):
        started = time.perf_counter()
        self.events = []
        now = packet.timestamp_ms
        actions={}
        for req in requests:
            if req.session_id != self.session_id or req.request_id in self.seen_requests:
                continue
            self.seen_requests.add(req.request_id)
            self.event("VOICE_REQUEST" if req.source=="voice" else "REQUEST", now,req.person_id,req.source,requested_at_ms=req.requested_at_ms,request_id=req.request_id,action=req.action,reason=req.reason)
            if req.action in {"PROTECT_PERSON","UNPROTECT_PERSON"} and req.person_id in self.registry.people:
                actions.setdefault(req.person_id,set()).add(req.action)
            elif req.action=="SHIELD_UNRESOLVED":
                self.mark_unresolved()
        try:
            observations = self.detectors.detect(packet.bgr, now)
            boxes, uncertain, full_shield = self._observe(observations, now, actions)
            shield = uncertain
            protected = protect_scene(packet.bgr) if full_shield else protect(packet.bgr, boxes)
            state = "SHIELDED" if shield else "NORMAL"
            mode = "SCENE_SHIELD" if full_shield else "FACE_SHIELD" if uncertain else "TARGETED"
            if self.unresolved:
                reason = "Earlier privacy request pending — unresolved located faces temporarily protected"
            else:
                reason = ("Head location uncertain — scene temporarily obscured" if full_shield else "Identity uncertain — temporarily blurring located faces") if uncertain else None
        except Exception as exc:
            # The only error output is an opaque slate; raw pixels are never the fallback.
            protected = protect(packet.bgr, shield=True)
            state, reason = "ERROR", "Vision processing failed: " + type(exc).__name__
            mode = "ERROR_SHIELD"
            self.event("ERROR",now,component="vision",error_type=type(exc).__name__)
        if state != self.output_state or mode != self.protection_mode:
            self.event("OUTPUT_RESUMED" if state=="NORMAL" else "OUTPUT_SHIELDED",now,reason=reason,protection_mode=mode)
        self.output_state = state
        self.protection_mode = mode
        height,width = packet.bgr.shape[:2]
        views = []
        for person in self.registry.people.values():
            track = next((t for t in self.tracks.values() if t.person_id==person.person_id and t.last_seen==now), None)
            temporary=bool(track and track.track_id in self.temporary_tracks and not person.protected)
            views.append(TrackView(person.person_id,"BLURRED" if person.protected else "UNBLURRED", "VISIBLE" if track else "OUT_OF_FRAME", "AMBIGUOUS" if track and (track.identity_uncertain or temporary) else "KNOWN", self._normalized(track.box,width,height) if track else None, track.progress if track else 0.,person.protection_source,person.last_seen_ms,track.track_id if track else None,temporary))
        for track in self.tracks.values():
            if track.person_id is None and track.last_seen==now:
                views.append(TrackView("pending_"+track.track_id,"PENDING", "VISIBLE", "PENDING", self._normalized(track.box,width,height),track.progress,None,now,track.track_id,track.track_id in self.temporary_tracks))
        return FrameResult(packet.frame_id,now,protected,views,self.events,state,(time.perf_counter()-started)*1000,reason,mode)

    @staticmethod
    def _normalized(box,width,height):
        x,y,w,h = box
        x0,y0=max(0.,x/width),max(0.,y/height)
        return [x0,y0,max(0.,min(1.,(x+w)/width)-x0),max(0.,min(1.,(y+h)/height)-y0)]

    def _observe(self, observations, now, actions=None):
        actions=actions or {}
        self.temporary_tracks=set()
        # Short-lived geometry is never the identity database.
        if any(t.pending_protect and now-t.last_seen>750 for t in self.tracks.values()):
            # A confirmed anonymous gesture cannot disappear with its motion track.
            self.mark_unresolved()
        self.tracks = {k:t for k,t in self.tracks.items() if now-t.last_seen <= 750}
        previous_visible = {p.person_id for p in self.registry.people.values() if p.visible}
        for p in self.registry.people.values():
            p.visible=False
        available = set(self.tracks)
        assignments=[]
        # Global greedy nearest pairs with explicit local ambiguity checks.
        pairs=sorted((distance(o.box,t.box), i,tid) for i,o in enumerate(observations) for tid,t in self.tracks.items())
        used_obs=set()
        for d,index,tid in pairs:
            if index in used_obs or tid not in available or d>.9:
                continue
            track=self.tracks[tid]; obs=observations[index]
            alternatives=[dd for dd,j,tt in pairs if j==index and tt in available and tt!=tid]
            other_observations=[dd for dd,j,tt in pairs if tt==tid and j not in used_obs and j!=index]
            ambiguity=bool((alternatives and min(alternatives)-d<.18) or (other_observations and min(other_observations)-d<.18))
            # A clear incompatible face cannot inherit a spatial track's privacy.
            if track.person_id and obs.feature is not None:
                score=similarity(obs.feature,self.registry.people[track.person_id])
                if score < self.threshold-.08:
                    others=[p for p in self.registry.people.values() if p.person_id!=track.person_id]
                    better=select(obs.feature,others,self.threshold,self.margin)
                    if better or d>.4 or now-track.last_seen>200:
                        continue
                    # Preserve a continuous geometry track through weak/profile
                    # features, but freeze identity updates and shield its face.
                    ambiguity=True
            track.identity_uncertain=ambiguity
            assignments.append((track,obs)); available.remove(tid); used_obs.add(index)
        for i,obs in enumerate(observations):
            if i not in used_obs:
                tid="t_"+uuid.uuid4().hex[:8]
                track=Track(tid,obs.box,now,hold=Hold(self.hold_ms))
                self.tracks[tid]=track
                assignments.append((track,obs))
        claimed=set()
        boxes=[]; uncertain=False; face_uncertain=False; gallery_updates=[]; gesture_candidate=False
        # Existing IDs reserve one-to-one assignment before pending matches.
        occupied={t.person_id for t,o in assignments if t.person_id}
        protected_people=[p for p in self.registry.people.values() if p.protected]
        resolved_protected={t.person_id:o for t,o in assignments
            if t.person_id and self.registry.people[t.person_id].protected
            and not t.identity_uncertain and o.quality and o.feature is not None
            and select(o.feature,self.registry.people.values(),self.threshold,self.margin)==t.person_id}
        for track,obs in assignments:
            track.box=obs.box; track.last_seen=now; track.observations+=1
            if obs.face_detected or obs.head_reliable:
                track.reliable_box=obs.box; track.reliable_at=now
            raw_cover, palm_center=bilateral_cover(obs.eyes if obs.face_detected or obs.head_reliable else None,obs.hands,obs.box[2])
            gesture_candidate=gesture_candidate or raw_cover
            # Temporal evidence may establish an objection while identity is
            # unresolved. It must not attach that objection to an uncertain ID.
            peace,peace_center=peace_sign(obs.hands,obs.eyes,obs.box[2])
            conflict=raw_cover and peace
            valid=raw_cover and not conflict and not obs.ownership_uncertain
            before=track.progress
            # Relative to eye anchor so whole-person motion doesn't look like rubbing.
            anchor=obs.eyes.mean(axis=0) if obs.eyes is not None else np.zeros(2)
            relative_center=palm_center-anchor if palm_center is not None else None
            track.progress=track.hold.update(valid,relative_center,now,obs.box[2])
            if track.progress==0:track.owner_uncertain_confirmed=False
            if before==0 and track.progress>0:
                self.event("GESTURE_STARTED",now,track.person_id)
            if before>0 and track.progress==0:
                self.event("GESTURE_CANCELLED",now,track.person_id)
            if track.person_id is None:
                if obs.quality and not track.identity_uncertain and now-track.enrollment_last_ms>=200:
                    if track.enrollment_samples and float(np.dot(obs.feature,track.enrollment_samples[-1]))<self.threshold+.10:
                        track.enrollment_samples=[]
                    track.enrollment_samples.append(obs.feature.copy())
                    track.enrollment_samples=track.enrollment_samples[-5:]
                    track.enrollment_last_ms=now
                candidates=[p for p in self.registry.people.values() if p.person_id not in occupied and p.person_id not in claimed]
                match=select(obs.feature,candidates,self.threshold,self.margin) if obs.feature is not None and not track.identity_uncertain else None
                if match:
                    track.agreements=track.agreements+1 if match==track.match_id else 1
                    track.match_id=match
                    if track.agreements>=3:
                        track.person_id=match
                        occupied.add(match)
                        self.event("PERSON_REIDENTIFIED",now,match,restored_privacy="BLURRED" if self.registry.people[match].protected else "UNBLURRED")
                else:
                    track.match_id=None; track.agreements=0
                    # A newcomer can be distinguished safely when every protected
                    # identity is independently located and matched in this frame.
                    # Missing protected people never become new unblurred IDs.
                    distinct_newcomer=bool(protected_people and obs.quality and obs.feature is not None
                        and all(p.person_id in resolved_protected and p.gallery
                            and similarity(obs.feature,p)<self.threshold-.15
                            and iou(obs.box,resolved_protected[p.person_id].box)<.1
                            and distance(obs.box,resolved_protected[p.person_id].box)>.9
                            for p in protected_people))
                    plausible_existing=bool(obs.feature is not None and candidates and rank(obs.feature,candidates)[0][0]>=self.threshold-self.margin)
                    # Pending privacy masks pixels; it must not prevent learning
                    # a reliable identity needed for an owned reveal. Enroll it
                    # protected, never by assigning/overwriting a missing ID.
                    # Check ALL galleries, including occupied IDs, to avoid
                    # enrolling a second view of an already visible person.
                    pending_enrollment=bool(self.unresolved and obs.quality and obs.feature is not None
                        and all(similarity(obs.feature,p)<self.threshold-self.margin
                            for p in self.registry.people.values()))
                    if obs.quality and len(track.enrollment_samples)>=3 and (not self.registry.has_protected or distinct_newcomer or pending_enrollment) and not track.identity_uncertain and not plausible_existing and (not self.unresolved or pending_enrollment):
                        person=self.registry.create(); track.person_id=person.person_id
                        person.gallery=[f.copy() for f in track.enrollment_samples]
                        person.last_sample_ms=now
                        if self.unresolved:
                            self.activate(person.person_id,"pending_request",now)
                        self.event("PERSON_SEEN",now,person.person_id)
                if valid and track.progress>=1 and not track.owner_uncertain_confirmed:
                    track.pending_protect=True
            if track.person_id:
                if track.person_id in claimed:
                    uncertain=True; boxes.append(head_box(obs.box)); continue
                claimed.add(track.person_id)
                person=self.registry.people[track.person_id]
                person.visible=True; person.last_seen_ms=now
                voice_actions=actions.get(person.person_id,set())
                reveal_ok=bool(peace and not conflict and not obs.ownership_uncertain
                    and not track.identity_uncertain and obs.quality and obs.feature is not None
                    and select(obs.feature,self.registry.people.values(),self.threshold,self.margin)==person.person_id)
                reveal_now=track.reveal.update(reveal_ok,peace_center-anchor if peace_center is not None else None,now,obs.box[2])
                wants_protect=(valid and track.progress>=1) or track.pending_protect or "PROTECT_PERSON" in voice_actions
                wants_reveal=reveal_now or "UNPROTECT_PERSON" in voice_actions
                cue_conflict=conflict or (wants_reveal and (raw_cover or wants_protect)) or (peace and "PROTECT_PERSON" in voice_actions)
                if cue_conflict:
                    track.pending_protect=False
                    track.hold=Hold(self.hold_ms); track.progress=0
                elif wants_reveal and not track.identity_uncertain and not obs.ownership_uncertain:
                    # Voice target was resolved across the utterance; recheck the
                    # current face before revealing after network delay.
                    if obs.quality and obs.feature is not None and select(obs.feature,self.registry.people.values(),self.threshold,self.margin)==person.person_id:
                        self.deactivate(person.person_id,"gesture" if reveal_now else "voice",now)
                elif wants_protect:
                    if track.owner_uncertain_confirmed:
                        pass # ambiguous hand ownership is a retained global request
                    elif track.identity_uncertain and "PROTECT_PERSON" not in voice_actions:
                        track.pending_protect=True
                    else:
                        self.activate(person.person_id,"voice" if "PROTECT_PERSON" in voice_actions else "gesture",now)
                        track.pending_protect=False
                # Never contaminate gallery during cover, a crossing, or weak alignment.
                if obs.quality and not valid and not track.identity_uncertain and now-person.last_sample_ms>=200:
                    if not person.gallery or similarity(obs.feature,person)>=self.threshold:
                        gallery_updates.append((person,obs.feature.copy()))
                if person.protected or track.identity_uncertain:
                    boxes.append(head_box(obs.box))
                if person.protected and not (obs.face_detected or obs.head_reliable):
                    if track.reliable_box is not None and now-track.reliable_at<=750:
                        x,y,w,h=track.reliable_box; a,b,c,d=obs.box
                        union=(min(x,a),min(y,b),max(x+w,a+c)-min(x,a),max(y+h,b+d)-min(y,b))
                        boxes.append(head_box(union))
                    face_uncertain=True
                    self.temporary_tracks.add(track.track_id)
            elif self.registry.has_protected or track.pending_protect or valid:
                boxes.append(head_box(obs.box))
                face_uncertain=True
                self.temporary_tracks.add(track.track_id)
            # A hand near a face is tracking uncertainty, not a privacy request.
            # Only bilateral eye coverage makes uncertain ownership relevant to
            # rendering; head reliability/identity safety is checked separately.
            if obs.ownership_uncertain and (raw_cover or track.pending_protect):
                uncertain=True
            if track.identity_uncertain:
                # Identity ambiguity has a located candidate: mask that face,
                # not the room or an independently resolved neighboring person.
                face_uncertain=True
                self.temporary_tracks.add(track.track_id)
        for pid,requested in actions.items():
            if requested=={"PROTECT_PERSON"} and not self.registry.people[pid].visible:
                self.activate(pid,"voice",now)
        current_visible={p.person_id for p in self.registry.people.values() if p.visible}
        for pid in previous_visible-current_visible:
            self.event("PERSON_LEFT",now,pid)
        # Face-only absence policy: an out-of-view preference remains in the
        # registry but is not itself evidence of a face in this image. Unknown
        # returning candidates are already masked above, before identity resolves.
        # Actual located-head uncertainty and processing errors still fail closed.
        # Pending requests remain pending, but absence has no pixels to mask.
        # A verified explicit reveal exempts only that ID from this pending batch.
        if self.unresolved:
            for t,o in assignments:
                if t.person_id not in self.unresolved_exemptions or t.identity_uncertain:
                    boxes.append(head_box(o.box))
                    self.temporary_tracks.add(t.track_id)
                    face_uncertain=True
        full_shield=False
        if uncertain:
            boxes.extend(head_box(o.box) for o in observations)
            self.temporary_tracks.update(t.track_id for t,o in assignments)
            face_uncertain=bool(observations)
            uncertain=False
        if not uncertain and not face_uncertain and not self.unresolved:
            for person,feature in gallery_updates:
                person.gallery.append(feature); person.gallery=person.gallery[-5:]
                person.last_sample_ms=now
        privacy_relevant=self.registry.has_protected or self.unresolved or gesture_candidate or any(t.pending_protect or t.progress>0 for t in self.tracks.values())
        if not privacy_relevant:
            # Preserve uncertain IDs and frozen galleries, but do not invent a
            # privacy objection from ordinary motion, face touch or enrollment.
            boxes=[]; uncertain=False; face_uncertain=False; full_shield=False
            self.temporary_tracks=set()
        return boxes,uncertain or face_uncertain,full_shield

    def close(self):
        self.detectors.close()
