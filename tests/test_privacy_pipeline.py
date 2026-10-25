"""Controlled observations verify policy, not real-world detector accuracy."""
import asyncio
import queue
import threading
import time
from dataclasses import asdict
import av
import cv2
import numpy as np
import pytest
from shared.contracts import FramePacket, PrivacyRequest
from vision.detectors import Observation
from vision.engine import VisionEngine
from vision.gestures import bilateral_cover, Hold
from vision.identity import normalize
from backend.recording import PairedRecorder
from backend.runtime import Runtime
from backend.meta.intent import Intent, Spark, fixed_phrase

class Detector:
    def __init__(self): self.observations=[]; self.fail=False
    def detect(self,image,stamp):
        if self.fail: raise RuntimeError("deliberate detector failure")
        return self.observations
    def close(self): pass

def feature(index):
    vector=np.zeros(128,np.float32); vector[index]=1
    return vector

def face(x=120,index=0):
    return Observation((x,120,100,120),eyes=np.array([[x+30,160],[x+70,160]],np.float32),feature=feature(index),quality=True,face_detected=True)

def palm(x=120):
    # Open hand spanning both eye regions. Actual landmark detection is separate.
    return np.array([[x+50,200],[x+18,190],[x+5,168],[x-5,150],[x-15,135],
        [x+15,135],[x+12,115],[x+10,95],[x+8,78],
        [x+40,128],[x+40,105],[x+40,80],[x+40,60],
        [x+65,132],[x+67,110],[x+70,90],[x+72,70],
        [x+85,145],[x+90,125],[x+95,105],[x+100,90]],np.float32)

def packet(stamp):
    image=np.random.default_rng(4).integers(0,256,(480,640,3),np.uint8)
    image.setflags(write=False)
    return FramePacket(stamp//100+1,stamp,image)

def enroll(engine,detector,obs,start=0):
    detector.observations=obs
    result=None
    for stamp in range(start,start+700,100): result=engine.step(packet(stamp),[])
    return result

def request(engine,pid,stamp=800,action="PROTECT_PERSON",session=None):
    return PrivacyRequest("q"+str(stamp),session or engine.session_id,"voice",pid,stamp,action)

def test_gesture_latches_target_and_original_unchanged():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face(),face(400,1)])
    ids=[t.person_id for t in result.tracks]
    assert len(ids)==2 and len(set(ids))==2
    first,second=ids
    obs=face(); obs.hands=[palm()]; obs.quality=False; obs.feature=None
    assert bilateral_cover(obs.eyes,obs.hands,100)[0]
    detector.observations=[obs,face(400,1)]
    for stamp in range(700,1800,100):
        source=packet(stamp); original=source.bgr.copy()
        result=engine.step(source,[])
        assert np.array_equal(original,source.bgr)
    assert engine.registry.people[first].protected
    assert not engine.registry.people[second].protected
    detector.observations=[face(),face(400,1)]
    result=engine.step(packet(1900),[])
    assert result.output_state=="NORMAL"
    assert not np.array_equal(result.protected_bgr[120:240,120:220],packet(1900).bgr[120:240,120:220])
    assert np.array_equal(result.protected_bgr[120:240,400:500],packet(1900).bgr[120:240,400:500])

def test_return_restores_same_id_without_a_raw_pending_frame():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face(),face(400,1)])
    pid=result.tracks[0].person_id
    engine.step(packet(800),[request(engine,pid)])
    gallery=[sample.copy() for sample in engine.registry.people[pid].gallery]
    detector.observations=[face(400,1)]
    lost=engine.step(packet(1800),[])
    assert lost.output_state=="SHIELDED"  # expired track must first re-identify
    engine.step(packet(1900),[])
    located=engine.step(packet(2000),[])
    assert located.output_state=="NORMAL"
    assert np.array_equal(located.protected_bgr,packet(2000).bgr)
    detector.observations=[face(300,0),face(500,1)]
    results=[engine.step(packet(t),[]) for t in [3000,3100,3200,3300]]
    assert all(not np.array_equal(r.protected_bgr[120:240,300:400],packet(r.timestamp_ms).bgr[120:240,300:400]) for r in results)
    assert any(e.type=="PERSON_REIDENTIFIED" and e.person_id==pid for r in results for e in r.events)
    assert [t for t in results[-1].tracks if t.person_id==pid][0].privacy=="BLURRED"
    assert len({t.person_id for t in results[-1].tracks if t.visibility=="VISIBLE"})==2
    assert all(np.array_equal(a,b) for a,b in zip(gallery,engine.registry.people[pid].gallery[:len(gallery)]))

def test_unknown_after_protection_stays_shielded_without_gallery_poisoning():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face()]); pid=result.tracks[0].person_id
    engine.step(packet(800),[request(engine,pid)])
    before=[f.copy() for f in engine.registry.people[pid].gallery]
    detector.observations=[face(450,3)]
    for t in range(2000,4000,100):
        result=engine.step(packet(t),[])
        assert result.output_state=="SHIELDED"
    assert len(engine.registry.people)==1
    assert all(np.array_equal(a,b) for a,b in zip(before,engine.registry.people[pid].gallery))

def test_newcomer_enrolls_unprotected_only_when_all_protected_people_are_resolved():
    detector=Detector();engine=VisionEngine('s_newcomer',detector)
    first=enroll(engine,detector,[face()]);pid=first.tracks[0].person_id
    engine.step(packet(700),[request(engine,pid,700)])
    detector.observations=[face(),face(400,1)]
    for t in range(800,1500,100):result=engine.step(packet(t),[])
    assert len(engine.registry.people)==2
    newcomer=next(p for p in engine.registry.people.values() if p.person_id!=pid)
    assert not newcomer.protected and engine.registry.people[pid].protected
    assert np.array_equal(result.protected_bgr[120:240,400:500],packet(1400).bgr[120:240,400:500])
    assert not np.array_equal(result.protected_bgr[140:220,130:210],packet(1400).bgr[140:220,130:210])
    # A third face cannot be enrolled merely because B is known while A is absent.
    detector.observations=[face(400,1),face(250,3)]
    for t in range(2500,4000,100):result=engine.step(packet(t),[])
    assert len(engine.registry.people)==2
    assert any(t.person_id.startswith('pending_') and t.temporary_protection for t in result.tracks)
    assert np.array_equal(result.protected_bgr[120:240,400:500],packet(3900).bgr[120:240,400:500])

def test_duplicate_protected_face_cannot_enroll_as_unprotected_newcomer():
    detector=Detector();engine=VisionEngine('s_duplicate',detector)
    pid=enroll(engine,detector,[face()]).tracks[0].person_id
    engine.step(packet(700),[request(engine,pid,700)])
    detector.observations=[face(),face(400,0)]
    for t in range(800,2000,100):engine.step(packet(t),[])
    assert len(engine.registry.people)==1 and engine.registry.people[pid].protected

def test_weak_features_do_not_fragment_continuous_identity():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face()]); pid=result.tracks[0].person_id
    before=[sample.copy() for sample in engine.registry.people[pid].gallery]
    detector.observations=[face(index=3)]
    for stamp in range(700,1800,100):
        source=packet(stamp); result=engine.step(source,[])
        assert len(engine.registry.people)==1
        assert result.tracks[0].person_id==pid
        assert result.tracks[0].identity_status=='AMBIGUOUS'
        assert not result.tracks[0].temporary_protection
        assert result.output_state=="NORMAL"
        assert np.array_equal(result.protected_bgr,source.bgr)
    assert all(np.array_equal(a,b) for a,b in zip(before,engine.registry.people[pid].gallery))
    result=engine.step(packet(1800),[request(engine,pid,1800)])
    assert result.output_state=='SHIELDED' and engine.registry.people[pid].protected

def test_incidental_hand_uncertainty_does_not_blur_unrelated_person():
    detector=Detector(); engine=VisionEngine('s_test',detector)
    initial=enroll(engine,detector,[face(),face(400,1)])
    pid=initial.tracks[0].person_id
    other=face(400,1); other.quality=False; other.feature=None
    other.ownership_uncertain=True; other.hands=[palm(400)+np.array([0,100],np.float32)]
    detector.observations=[face(),other]
    source=packet(800); result=engine.step(source,[])
    assert result.output_state=='NORMAL' and np.array_equal(result.protected_bgr,source.bgr)
    source=packet(900); result=engine.step(source,[request(engine,pid,900)])
    assert result.output_state=='NORMAL' and engine.registry.people[pid].protected
    assert not np.array_equal(result.protected_bgr[140:220,130:210],source.bgr[140:220,130:210])
    assert np.array_equal(result.protected_bgr[120:240,400:500],source.bgr[120:240,400:500])

def test_confirmed_cover_waits_for_identity_without_losing_request():
    detector=Detector(); engine=VisionEngine('s_test',detector)
    result=enroll(engine,detector,[face()]); pid=result.tracks[0].person_id
    ambiguous=face(index=3); ambiguous.hands=[palm()]
    detector.observations=[ambiguous]
    for stamp in range(700,1800,100):result=engine.step(packet(stamp),[])
    assert not engine.registry.people[pid].protected
    assert next(iter(engine.tracks.values())).pending_protect
    assert result.output_state=='SHIELDED'
    detector.observations=[face()]
    result=engine.step(packet(1900),[])
    assert engine.registry.people[pid].protected and result.output_state=='NORMAL'

def test_confirmed_unknown_hand_owner_is_not_assigned_to_known_face():
    detector=Detector(); engine=VisionEngine('s_test',detector)
    result=enroll(engine,detector,[face()]); pid=result.tracks[0].person_id
    ambiguous=face(); ambiguous.hands=[palm()]; ambiguous.ownership_uncertain=True; ambiguous.quality=False; ambiguous.feature=None
    detector.observations=[ambiguous]
    for stamp in range(700,1800,100):engine.step(packet(stamp),[])
    detector.observations=[face()]
    for stamp in [1900,2200,2300]:result=engine.step(packet(stamp),[])
    assert not engine.unresolved and not engine.registry.people[pid].protected
    assert result.output_state=='NORMAL'

def test_absent_protected_person_does_not_blur_known_neighbor_or_empty_scene():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face(),face(400,1)])
    pid=result.tracks[0].person_id
    engine.step(packet(800),[request(engine,pid)])
    detector.observations=[face(400,1)]
    engine.step(packet(1800),[]);engine.step(packet(1900),[])
    source=packet(2000); result=engine.step(source,[])
    assert result.output_state=="NORMAL"
    assert np.array_equal(result.protected_bgr,source.bgr)
    assert not next(t for t in result.tracks if t.visibility=="VISIBLE").temporary_protection
    detector.observations=[]
    empty=engine.step(packet(5000),[])
    assert empty.protection_mode=="TARGETED"
    assert np.array_equal(empty.protected_bgr,packet(5000).bgr)
    assert engine.registry.people[pid].protected
    # Ten seconds of absence must never reintroduce a global shield, whether
    # the scene is empty or a known unprotected neighbor remains visible.
    for stamp in range(6000,26300,100):
        detector.observations=[] if stamp<16100 else [face(400,1)]
        frame=packet(stamp); absent=engine.step(frame,[])
        assert absent.protection_mode not in {'SCENE_SHIELD','ERROR_SHIELD'}
        if stamp<16100 or stamp>=16400:
            assert np.array_equal(absent.protected_bgr,frame.bgr)
        assert engine.registry.people[pid].protected
    # Even when an unrecognized return is not yet assigned to the saved ID,
    # its located face is masked on the first frame; the room stays unchanged.
    detector.observations=[face(300,3)]
    returned=engine.step(packet(27000),[])
    assert returned.protection_mode=='FACE_SHIELD'
    assert not np.array_equal(returned.protected_bgr[140:220,310:390],packet(27000).bgr[140:220,310:390])
    assert np.array_equal(returned.protected_bgr[:40],packet(27000).bgr[:40])

def test_failure_slate_and_session_fencing():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face()]); pid=result.tracks[0].person_id
    stale=request(engine,pid,session="s_old")
    engine.step(packet(800),[stale])
    assert not engine.registry.people[pid].protected
    detector.fail=True
    source=packet(900); original=source.bgr.copy()
    result=engine.step(source,[])
    assert result.output_state=="ERROR" and not np.array_equal(result.protected_bgr,source.bgr)
    assert np.array_equal(original,source.bgr)
    assert np.std(result.protected_bgr[0:50])==pytest.approx(np.std(np.full((50,640,3),(28,23,20),np.uint8)))

def test_natural_face_blur_suppresses_details_and_preserves_surroundings():
    from vision.rendering import protect,head_box
    source=packet(0).bgr
    result=protect(source,[head_box((200,150,100,130))])
    # Face center loses fine detail; outside the head and the oval corners stay intact.
    assert np.std(result[170:220,220:280])<12
    assert np.array_equal(result[5:30,5:30],source[5:30,5:30])
    x,y,w,h=head_box((200,150,100,130))
    assert np.array_equal(result[int(y):int(y)+3,int(x):int(x)+3],source[int(y):int(y)+3,int(x):int(x)+3])
    assert not np.shares_memory(result,source)

def test_ambiguous_voice_remains_shielded_and_deduplicates():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    enroll(engine,detector,[face()])
    unresolved=request(engine,None,action="SHIELD_UNRESOLVED")
    result=engine.step(packet(800),[unresolved,unresolved])
    assert result.output_state=="SHIELDED" and len([e for e in result.events if e.type=="VOICE_REQUEST"])==1
    assert result.protection_mode=="FACE_SHIELD"
    assert result.tracks[0].temporary_protection
    assert np.array_equal(result.protected_bgr[:40],packet(800).bgr[:40])
    assert not np.array_equal(result.protected_bgr[140:220,130:210],packet(800).bgr[140:220,130:210])
    assert engine.step(packet(900),[]).protection_mode=="FACE_SHIELD"
    assert not any(p.protected for p in engine.registry.people.values())

def test_unresolved_voice_keeps_empty_scene_clear_and_masks_returning_faces():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    enroll(engine,detector,[face()])
    detector.observations=[]
    result=engine.step(packet(800),[request(engine,None,action="SHIELD_UNRESOLVED")])
    assert result.protection_mode=="TARGETED"
    detector.observations=[face()]
    result=engine.step(packet(900),[])
    assert result.protection_mode=="FACE_SHIELD"
    assert engine.unresolved  # no quiet/speech timeout silently revokes the request
    assert np.array_equal(result.protected_bgr[:40],packet(900).bgr[:40])

def test_protected_person_cannot_use_an_unreliable_collapsed_head_box():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face()]); pid=result.tracks[0].person_id
    engine.step(packet(800),[request(engine,pid)])
    detector.observations=[Observation((150,150,28,30),eyes=np.array([[155,160],[158,160]],np.float32))]
    result=engine.step(packet(900),[])
    assert result.protection_mode=="FACE_SHIELD"
    assert not np.array_equal(result.protected_bgr[140:220,140:200],packet(900).bgr[140:220,140:200])
    assert np.array_equal(result.protected_bgr[:40],packet(900).bgr[:40])

def test_shield_recovery_preserves_legitimate_preference_identity_and_gallery():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face(),face(400,1)])
    pid=result.tracks[0].person_id
    engine.step(packet(800),[request(engine,pid)])
    gallery=[v.copy() for v in engine.registry.people[pid].gallery]
    detector.observations=[]
    result=engine.step(packet(2000),[request(engine,None,2000,"SHIELD_UNRESOLVED")])
    assert result.protection_mode=="TARGETED"
    detector.observations=[face(260,0),face(450,1)]
    results=[engine.step(packet(t),[]) for t in [3000,3100,3200,3300]]
    assert all(r.protection_mode=="FACE_SHIELD" for r in results)
    restored=next(t for t in results[-1].tracks if t.person_id==pid)
    assert restored.visibility=="VISIBLE" and restored.privacy=="BLURRED"
    assert engine.unresolved and engine.registry.people[pid].protected
    assert len(engine.registry.people)==2
    assert all(np.array_equal(a,b) for a,b in zip(gallery,engine.registry.people[pid].gallery))
    for result in results:
        source=packet(result.timestamp_ms).bgr
        assert np.array_equal(result.protected_bgr[:40],source[:40])
        assert not np.array_equal(result.protected_bgr[145:215,280:340],source[145:215,280:340])
    # A genuine detector error blocks pixels; resumption retains every preference.
    detector.fail=True
    assert engine.step(packet(3400),[]).protection_mode=="ERROR_SHIELD"
    detector.fail=False
    assert engine.step(packet(3500),[]).protection_mode=="FACE_SHIELD"
    assert engine.registry.people[pid].protected and engine.unresolved

def test_crossing_cannot_attach_same_identity_twice():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face(),face(400,1)])
    pid=result.tracks[0].person_id
    engine.step(packet(800),[request(engine,pid)])
    detector.observations=[face(230,0),face(240,1)]
    result=engine.step(packet(1000),[])
    ids=[t.person_id for t in result.tracks if t.visibility=="VISIBLE"]
    assert len(ids)==len(set(ids))
    assert result.output_state=="SHIELDED" or all(t.privacy=="BLURRED" or t.person_id!=pid for t in result.tracks)

def test_hold_rejects_brief_single_eye_hair_and_motion():
    eyes=face().eyes
    assert bilateral_cover(eyes,[palm()],100)[0]
    shifted=palm()+np.array([0,-110],np.float32)
    assert not bilateral_cover(eyes,[shifted],100)[0]
    one=palm().copy(); one[:,0]=120+(one[:,0]-120)*.3
    assert not bilateral_cover(eyes,[one],100)[0]
    hold=Hold()
    for t in [0,100,200,300]: assert hold.update(True,np.zeros(2),t,100)<1
    assert hold.update(False,None,500,100)==0
    # Large landmark dropout can't complete an 800 ms hold.
    hold=Hold(); hold.update(True,np.zeros(2),0,100)
    assert hold.update(True,np.zeros(2),1000,100)==0
    for t in range(1100,1900,100): progress=hold.update(True,np.zeros(2),t,100)
    assert progress==1

def test_two_hands_joint_coverage():
    left=palm().copy(); left[:,0]=120+(left[:,0]-120)*.55
    right=left+np.array([50,0],np.float32)
    assert not bilateral_cover(face().eyes,[left],100)[0]
    assert bilateral_cover(face().eyes,[left,right],100)[0]

def test_confirmed_anonymous_cover_cannot_expire_into_exposure():
    detector=Detector(); engine=VisionEngine("s_test",detector)
    obs=face(); obs.feature=None; obs.quality=False; obs.hands=[palm()]
    detector.observations=[obs]
    for t in range(0,1100,100):
        result=engine.step(packet(t),[])
    assert any(t.pending_protect for t in engine.tracks.values())
    detector.observations=[]
    assert engine.step(packet(2200),[]).output_state=="NORMAL"
    assert engine.unresolved
    detector.observations=[face(450,1)]
    assert engine.step(packet(2300),[]).output_state=="SHIELDED"

def decode(path):
    with av.open(str(path)) as container:
        return [(float(f.pts*f.time_base),f.to_ndarray(format="bgr24")) for f in container.decode(video=0)]

def test_paired_live_recording_variable_timestamps_and_latch(tmp_path):
    detector=Detector(); engine=VisionEngine("s_test",detector)
    result=enroll(engine,detector,[face()]); pid=result.tracks[0].person_id
    for sequence in range(2):
        recorder=PairedRecorder(tmp_path,640,480,"s_test")
        start=1000+sequence*2000
        for index,delta in enumerate([0,70,150,310,410,530]):
            source=packet(start+delta)
            result=engine.step(source,[request(engine,pid,start)] if sequence==0 and index==0 else [])
            recorder.write(source.bgr,result.protected_bgr,source.timestamp_ms)
        summary=recorder.close()
        assert summary["status"]=="FINALIZED"
        original=decode(recorder.directory/"original.mp4"); protected=decode(recorder.directory/"protected.mp4")
        assert len(original)==len(protected)==6
        assert [x[0] for x in original]==[x[0] for x in protected]
        assert original[-1][0]==pytest.approx(.53,abs=.002)
        assert np.mean(np.abs(original[0][1].astype(float)-protected[0][1]))>1
        assert engine.registry.people[pid].protected

def test_voice_validation_and_provider_failure(monkeypatch):
    assert fixed_phrase("Blur me!").intent=="PROTECT_PERSON"
    assert fixed_phrase("Don't blur me") is None
    with pytest.raises(Exception):
        Intent.model_validate({"intent":"UNBLUR","target_kind":"SELF","target_description":None,"needs_clarification":False})
    monkeypatch.delenv("MODEL_API_KEY",raising=False)
    intent,route=asyncio.run(Spark().classify("Would you keep my face out of this?"))
    assert intent.intent=="NEEDS_CLARIFICATION" and intent.needs_clarification

class FixtureCamera:
    opens=0
    def __init__(self,origin):
        self.origin=origin; self.frames=queue.Queue(maxsize=1); self.captured=0; self.dropped=0; self.error=None; self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self):
        FixtureCamera.opens+=1; self.thread.start()
    def run(self):
        while not self.stop.wait(.04):
            if self.error:continue
            self.captured+=1
            source=FramePacket(self.captured,int((time.monotonic()-self.origin)*1000),packet(0).bgr)
            try: self.frames.put_nowait(source)
            except queue.Full: self.dropped+=1
    def close(self):
        self.stop.set(); self.thread.join(1)
        return not self.thread.is_alive()

def wait_until(predicate,timeout=5):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.02)
    raise AssertionError("Timed out")

def test_upgrade_snapshot_retains_preferences_and_never_guesses_missing_gallery(tmp_path):
    detector=Detector(); detector.observations=[face()]
    runtime=Runtime(lambda sid:VisionEngine(sid,detector),FixtureCamera,tmp_path)
    runtime.start(recovery={"session_id":"s_previous","timestamp_ms":30000,"unresolved":True,
        "tracks":[{"person_id":"p_001","privacy":"BLURRED","protection_source":"gesture","last_seen_ms":29000}]})
    try:
        wait_until(lambda:runtime.state()["status"]=="RUNNING")
        assert runtime.session_id=="s_previous" and runtime.now_ms()>=30000
        assert runtime.engine.registry.people['p_001'].protected and runtime.engine.unresolved
        assert runtime.state()['unresolved_privacy_request'] is True
        assert runtime.state()['protection_mode']=='FACE_SHIELD'
        assert not any(t['person_id']=='p_001' and t['visibility']=='VISIBLE' for t in runtime.state()['tracks'])
    finally:runtime.end()

def test_private_gallery_recovery_preserves_id_and_absence_policy(tmp_path):
    detector=Detector()
    runtime=Runtime(lambda sid:VisionEngine(sid,detector),FixtureCamera,tmp_path)
    runtime.start(recovery={'session_id':'s_gallery','timestamp_ms':30000,
        'unresolved_privacy_request':False,
        'tracks':[{'person_id':'p_001','privacy':'BLURRED','protection_source':'gesture','last_seen_ms':29000}],
        'identity_galleries':{'p_001':[feature(0).tolist()]}})
    try:
        wait_until(lambda:runtime.state()['status']=='RUNNING')
        assert not runtime.engine.unresolved
        assert runtime.engine.registry.people['p_001'].protected
        assert runtime.state()['protection_mode']=='TARGETED'
        assert 'identity_galleries' not in runtime.state()
        detector.observations=[face()]
        wait_until(lambda:any(t['person_id']=='p_001' and t['visibility']=='VISIBLE' for t in runtime.state()['tracks']))
        assert next(t for t in runtime.state()['tracks'] if t['person_id']=='p_001')['privacy']=='BLURRED'
    finally:runtime.end()

def test_legacy_snapshot_false_flag_cannot_revoke_unknown_pending_privacy(tmp_path):
    detector=Detector(); detector.observations=[face()]
    runtime=Runtime(lambda sid:VisionEngine(sid,detector),FixtureCamera,tmp_path)
    runtime.start(recovery={'session_id':'s_legacy','timestamp_ms':50000,'tracks':[],
        'unresolved':False,'reason':'Privacy request pending — scene temporarily obscured'})
    try:
        wait_until(lambda:runtime.state().get('protection_mode')=='FACE_SHIELD')
        assert runtime.engine.unresolved and runtime.state()['unresolved_privacy_request']
    finally:runtime.end()

def test_integrated_runtime_preview_record_reconnect_and_failure(tmp_path):
    detector=Detector(); detector.observations=[face()]
    runtime=Runtime(lambda sid:VisionEngine(sid,detector),FixtureCamera,tmp_path)
    before=FixtureCamera.opens
    runtime.start()
    try:
        wait_until(lambda:len(runtime.state()["tracks"])==1 and runtime.state()["tracks"][0]["identity_status"]=="KNOWN")
        sid=runtime.session_id; pid=runtime.state()["tracks"][0]["person_id"]
        for _ in range(5):
            runtime.state(); seq,jpeg=runtime.preview(); assert jpeg.startswith(b'\xff\xd8')
        assert runtime.session_id==sid and FixtureCamera.opens==before+1
        with pytest.raises(ValueError):runtime.start()
        runtime.command("recording_start")
        runtime.enqueue(request(runtime.engine,pid,runtime.now_ms()))
        wait_until(lambda:runtime.state()["tracks"][0]["privacy"]=="BLURRED")
        time.sleep(.2)
        detector.fail=True
        wait_until(lambda:runtime.state()["output_state"]=="ERROR")
        summary=runtime.command("recording_stop")
        assert summary["status"]=="FINALIZED"
        directory=runtime.recordings[summary["recording_id"]][0]
        original=decode(directory/"original.mp4"); protected=decode(directory/"protected.mp4")
        assert len(original)==len(protected)>2
        assert np.std(protected[-1][1][0:50])<10
        detector.fail=False
        runtime.command("recording_start"); time.sleep(.2)
        second=runtime.command("recording_stop")
        assert second["status"]=="FINALIZED"
        assert runtime.engine.registry.people[pid].protected
    finally:
        runtime.end()

def test_camera_interruption_retains_engine_privacy_and_unique_frame_ids(tmp_path):
    detector=Detector(); detector.observations=[face()]
    runtime=Runtime(lambda sid:VisionEngine(sid,detector),FixtureCamera,tmp_path)
    runtime.start()
    try:
        wait_until(lambda:any(t['identity_status']=='KNOWN' for t in runtime.state()['tracks']))
        engine=runtime.engine; camera=runtime.camera; sid=runtime.session_id
        pid=runtime.state()['tracks'][0]['person_id']
        runtime.enqueue(request(engine,pid,runtime.now_ms()))
        wait_until(lambda:engine.registry.people[pid].protected)
        gallery=[v.copy() for v in engine.registry.people[pid].gallery]
        runtime.command('recording_start'); time.sleep(.15)
        camera.error='Controlled capture interruption'
        wait_until(lambda:runtime.state()['status']=='RECOVERING_SOURCE')
        assert runtime.state()['protection_mode']=='ERROR_SHIELD'
        assert runtime.engine is engine and engine.registry.people[pid].protected
        wait_until(lambda:runtime.camera is not camera and runtime.state()['output_state']=='NORMAL',timeout=8)
        assert not camera.thread.is_alive() and runtime.session_id==sid
        assert runtime.engine is engine and engine.registry.people[pid].protected
        assert all(np.array_equal(a,b) for a,b in zip(gallery,engine.registry.people[pid].gallery[:len(gallery)]))
        summary=runtime.command('recording_stop')
        directory=runtime.recordings[summary['recording_id']][0]
        import json
        rows=[json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
        ids=[row['frame_id'] for row in rows]
        assert len(ids)==len(set(ids)) and ids==sorted(ids)
        a,b=decode(directory/'original.mp4'),decode(directory/'protected.mp4')
        assert len(a)==len(b)>2 and [f[0] for f in a]==[f[0] for f in b]
        assert all(a[i][0]<a[i+1][0] for i in range(len(a)-1))
    finally:runtime.end()

def test_deployment_shutdown_saves_authoritative_privacy_clock_and_final_counter(monkeypatch,tmp_path):
    detector=Detector(); detector.observations=[face()]
    runtime=Runtime(lambda sid:VisionEngine(sid,detector),FixtureCamera,tmp_path/'recordings')
    path=tmp_path/'private/recovery.json'
    monkeypatch.setenv('CONSENTLENS_RECOVERY_SNAPSHOT',str(path))
    runtime.start()
    wait_until(lambda:any(t['identity_status']=='KNOWN' for t in runtime.state()['tracks']))
    pid=runtime.state()['tracks'][0]['person_id']
    runtime.enqueue(request(runtime.engine,pid,runtime.now_ms()))
    wait_until(lambda:runtime.engine.registry.people[pid].protected)
    camera=runtime.camera; origin=runtime.origin
    runtime.end(for_deployment=True)
    import json
    saved=json.loads(path.read_text())
    assert not camera.thread.is_alive()
    assert saved['capture_sequence_floor']==camera.captured
    assert saved['clock_origin_monotonic']==origin
    assert saved['tracks'][0]['privacy']=='BLURRED' and saved['tracks'][0]['protection_source']=='voice'
    assert path.stat().st_mode & 0o777==0o600
