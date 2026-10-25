"""Controlled geometry/timing checks; not physical gesture accuracy claims."""
import numpy as np
import pytest
from vision.gestures import bilateral_cover, Hold, DeliberateAction, peace_sign
from vision.engine import VisionEngine
from test_privacy_pipeline import face,palm,Detector,enroll,packet
from test_reversible import peace


def half_hands():
    left=palm();left[:,0]=120+(left[:,0]-120)*.55
    return left,left+np.array([50,0],np.float32)

@pytest.mark.parametrize('mirror',[False,True])
def test_both_eye_regions_one_or_two_hands(mirror):
    eyes=face().eyes.copy();hands=[palm(),*half_hands()]
    if mirror:
        eyes[:,0]=640-eyes[:,0]
        for h in hands:h[:,0]=640-h[:,0]
    assert bilateral_cover(eyes,[hands[0]],100)[0]
    assert not bilateral_cover(eyes,[hands[1]],100)[0]
    assert not bilateral_cover(eyes,[hands[2]],100)[0]
    assert bilateral_cover(eyes,hands[1:],100)[0]
    v=peace()
    if mirror:v[:,0]=640-v[:,0]
    assert peace_sign([v],eyes,100)[0]


def test_one_eye_collapsed_missing_and_nonfinite_geometry_rejected():
    eyes=face().eyes.copy()
    for invalid in [None,np.array([[150,160],[150,160]]),np.array([[150,160],[155,160]]),eyes*np.nan]:
        assert not bilateral_cover(invalid,[palm()],100)[0]
    assert not bilateral_cover(eyes,[palm()*np.nan],100)[0]
    for delta in [[0,-110],[0,-65],[0,85],[130,0]]:
        assert not bilateral_cover(eyes,[palm()+delta],100)[0] # hair/forehead/cheek/side


def test_one_eye_touches_cannot_accumulate_a_bilateral_hold():
    hold=Hold();left,right=half_hands();eyes=face().eyes
    for t in range(0,3000,50):
        hands=[left] if t%200<100 else [right]
        valid,center=bilateral_cover(eyes,hands,100)
        assert hold.update(valid,center,t,100)==0
    # Repeated short bilateral bursts separated by explicit one-eye frames.
    for t in range(3000,6000,50):
        valid=t%200!=0
        assert hold.update(valid,np.zeros(2) if valid else None,t,100)<1


def test_elapsed_time_hold_variable_fps_and_motion():
    hold=Hold()
    stamps=[0,44,96,151,224,310,401,496,555]
    for t in stamps[:-1]:assert hold.update(True,np.zeros(2),t,100)<1
    assert hold.update(True,np.zeros(2),stamps[-1],100)==1
    assert hold.update(False,None,590,100)==0
    assert hold.update(True,np.zeros(2),620,100)==0
    assert hold.update(True,np.zeros(2),1000,100)==0 # stale detection cannot count
    assert hold.update(True,np.array([30,0]),1050,100)==0 # rapid rubbing/sweep


def test_peace_separate_hold_and_rearm():
    reveal=DeliberateAction();hits=[]
    for t in range(0,1500,50):
        if reveal.update(True,np.zeros(2),t,100):hits.append(t)
    assert hits==[450]
    for t in [1500,1600,1700]:assert not reveal.update(False,None,t,100)
    assert not reveal.latched
    assert not reveal.update(True,np.zeros(2),1750,100)
    for t in range(1800,2201,50):
        hit=reveal.update(True,np.zeros(2),t,100)
        assert hit==(t==2200)


def test_wrong_owner_or_unreliable_eye_geometry_never_activates(monkeypatch):
    monkeypatch.delenv('GESTURE_HOLD_MS',raising=False)
    for uncertain,unreliable in [(True,False),(False,True)]:
        d=Detector();e=VisionEngine('s',d);pid=enroll(e,d,[face()]).tracks[0].person_id
        o=face();o.hands=[palm()];o.ownership_uncertain=uncertain
        o.face_detected=not unreliable;o.head_reliable=False
        d.observations=[o]
        for t in range(700,2000,50):e.step(packet(t),[])
        assert not e.registry.people[pid].protected


def test_quick_changed_gesture_has_no_shared_cooldown(monkeypatch):
    monkeypatch.delenv('GESTURE_HOLD_MS',raising=False)
    d=Detector();e=VisionEngine('s',d);pid=enroll(e,d,[face()]).tracks[0].person_id
    o=face();o.hands=[palm()];o.quality=False;o.feature=None;d.observations=[o]
    for t in range(700,1251,50):e.step(packet(t),[])
    assert e.registry.people[pid].protected
    d.observations=[face()];e.step(packet(1300),[])
    assert e.registry.people[pid].protected
    o=face();o.hands=[peace()];d.observations=[o]
    for t in range(1350,1801,50):e.step(packet(t),[])
    assert not e.registry.people[pid].protected


def test_slow_source_needs_time_and_four_current_valid_observations():
    hold=Hold()
    for t in [0,200,400]:assert hold.update(True,np.zeros(2),t,100)<1
    assert hold.update(True,np.zeros(2),600,100)==1
    reveal=DeliberateAction()
    assert [t for t in [0,200,400,600] if reveal.update(True,np.zeros(2),t,100)]==[600]
    # Two sparse samples are insufficient even if their elapsed time exceeds hold.
    hold=Hold(450)
    assert hold.update(True,np.zeros(2),0,100)<1
    assert hold.update(True,np.zeros(2),300,100)<1
    assert hold.update(True,np.zeros(2),600,100)<1
    assert hold.update(False,None,650,100)==0 # one-eye/non-cover still resets
    assert hold.update(True,np.zeros(2),700,100)==0
    assert hold.update(True,np.zeros(2),1200,100)==0 # stale gap still resets
