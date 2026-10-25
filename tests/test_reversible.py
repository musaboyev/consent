import numpy as np
from test_privacy_pipeline import Detector,face,palm,packet,enroll,request
from vision.engine import VisionEngine
from vision.gestures import peace_sign,DeliberateAction
from backend.meta.intent import fixed_phrase

def peace():
    # Beside face; index/middle straight, ring/little tips folded toward wrist.
    return np.array([[300,250],[280,240],[275,225],[280,215],[290,220],
        [285,210],[280,180],[275,155],[270,130],
        [305,205],[310,175],[315,150],[320,125],
        [320,215],[330,190],[328,220],[320,230],
        [335,225],[345,205],[340,230],[335,240]],np.float32)

def test_peace_configuration_and_debounce():
    assert peace_sign([peace()],face().eyes,100)[0]
    assert not peace_sign([palm()],face().eyes,100)[0]
    h=peace();h[16]=[340,120]
    assert not peace_sign([h],face().eyes,100)[0]
    hold=DeliberateAction()
    assert sum(hold.update(True,np.array([0,0]),t,100) for t in range(0,2000,100))==1
    for t in range(2000,2500,100):hold.update(False,None,t,100)
    assert sum(hold.update(True,np.array([0,0]),t,100) for t in range(2500,3600,100))==1

def test_two_cycles_and_latest_preference_return_with_neighbor_clear():
    d=Detector();e=VisionEngine('s',d);r=enroll(e,d,[face(),face(450,1)])
    a,b=[t.person_id for t in r.tracks]
    for start in [700,4000]:
        cover=face();cover.hands=[palm()];cover.feature=None;cover.quality=False
        d.observations=[cover,face(450,1)]
        for t in range(start,start+1100,100): e.step(packet(t),[])
        assert e.registry.people[a].protected and not e.registry.people[b].protected
        d.observations=[face(),face(450,1)]
        for t in range(start+1100,start+1500,100):e.step(packet(t),[])
        assert e.registry.people[a].protected
        v=face();v.hands=[peace()];d.observations=[v,face(450,1)]
        events=[]
        for t in range(start+1500,start+2800,100):events+=e.step(packet(t),[]).events
        assert not e.registry.people[a].protected
        assert sum(x.type=='PROTECTION_REMOVED' for x in events)==1
        d.observations=[face(),face(450,1)]
        for t in range(start+2800,start+3300,100):e.step(packet(t),[])
    d.observations=[]
    for t in range(8000,19000,100):
        r=e.step(packet(t),[])
        assert np.array_equal(r.protected_bgr,packet(t).bgr)
    d.observations=[face()]
    for t in range(19000,19600,100):r=e.step(packet(t),[])
    assert e.registry.people[a].visible and not e.registry.people[a].protected
    assert np.array_equal(r.protected_bgr,packet(19500).bgr)

def test_conflicting_voice_and_cover_cannot_reveal():
    d=Detector();e=VisionEngine('s',d);pid=enroll(e,d,[face()]).tracks[0].person_id
    e.step(packet(800),[request(e,pid)])
    cover=face();cover.hands=[palm()];d.observations=[cover]
    e.step(packet(900),[request(e,pid,900,'UNPROTECT_PERSON')])
    assert e.registry.people[pid].protected
    d.observations=[face()]
    e.step(packet(1000),[request(e,pid,1000,'UNPROTECT_PERSON')])
    assert not e.registry.people[pid].protected
    e.step(packet(1100),[request(e,pid,1100),request(e,pid,1101,'UNPROTECT_PERSON')])
    assert not e.registry.people[pid].protected

def test_reveal_requires_current_clear_identity_and_no_uncertain_request():
    d=Detector();e=VisionEngine('s',d);pid=enroll(e,d,[face()]).tracks[0].person_id
    e.step(packet(800),[request(e,pid)])
    d.observations=[]
    e.step(packet(900),[request(e,pid,900,'UNPROTECT_PERSON')])
    assert e.registry.people[pid].protected
    for text in ['Don\'t unblur me','What does "unblur me" mean?','He said unblur me','Can I say blur me?']:
        assert fixed_phrase(text) is None
    assert fixed_phrase('Unblur me.').intent=='UNPROTECT_PERSON'


def test_quotes_and_command_questions_never_queue_even_on_provider_failure():
    from backend.meta.intent import Intent,uncertain,should_queue_privacy
    for text in ['He said "hide my face"','What does blur me mean?','What happens if I say unblur me?']:
        assert not should_queue_privacy(uncertain(),text)
        for action in ['PROTECT_PERSON','UNPROTECT_PERSON']:
            assert not should_queue_privacy(Intent(intent=action,target_kind="SELF",target_description=None,needs_clarification=False),text)


def test_simultaneous_cover_and_peace_retains_either_existing_state():
    for protected in [False,True]:
        d=Detector();e=VisionEngine('s',d);pid=enroll(e,d,[face()]).tracks[0].person_id
        e.registry.people[pid].protected=protected
        both=face();both.hands=[palm(),peace()];d.observations=[both]
        for t in range(700,2500,100):e.step(packet(t),[])
        assert e.registry.people[pid].protected==protected

def test_peace_requires_resolved_owner_and_rejects_brief_touch():
    d=Detector();e=VisionEngine('s',d);pid=enroll(e,d,[face()]).tracks[0].person_id
    e.step(packet(800),[request(e,pid)])
    obs=face();obs.hands=[peace()];obs.ownership_uncertain=True;d.observations=[obs]
    for t in range(900,2100,100):e.step(packet(t),[])
    assert e.registry.people[pid].protected
    obs.ownership_uncertain=False
    for t in range(2100,2400,100):e.step(packet(t),[])
    d.observations=[face()]
    for t in range(2400,2800,100):e.step(packet(t),[])
    assert e.registry.people[pid].protected


def test_pending_request_never_blurs_absence_and_explicit_reveal_is_person_scoped():
    d=Detector();e=VisionEngine('s',d);r=enroll(e,d,[face(),face(450,1)])
    a,b=[t.person_id for t in r.tracks]
    e.mark_unresolved();d.observations=[]
    for t in range(800,11000,100):
        r=e.step(packet(t),[])
        assert np.array_equal(r.protected_bgr,packet(t).bgr)
        assert e.unresolved
    d.observations=[face(),face(450,1)]
    for t in range(11000,11500,100):e.step(packet(t),[])
    v=face();v.hands=[peace()];d.observations=[v,face(450,1)]
    for t in range(11500,12600,100):r=e.step(packet(t),[])
    assert e.unresolved and e.unresolved_exemptions=={a}
    assert np.array_equal(r.protected_bgr[140:220,140:200],packet(12500).bgr[140:220,140:200])
    assert not np.array_equal(r.protected_bgr[140:220,470:530],packet(12500).bgr[140:220,470:530])
    e.mark_unresolved()
    assert not e.unresolved_exemptions


def test_both_hands_and_camera_mirroring_have_identical_rules():
    from vision.gestures import bilateral_cover
    eyes=face().eyes
    for hand in [peace(),palm()]:
        original_peace=peace_sign([hand],eyes,100)[0]
        original_cover=bilateral_cover(eyes,[hand],100)[0]
        mirrored=hand.copy();mirrored[:,0]=640-mirrored[:,0]
        reflected=eyes.copy();reflected[:,0]=640-reflected[:,0]
        assert peace_sign([mirrored],reflected,100)[0]==original_peace
        assert bilateral_cover(reflected,[mirrored],100)[0]==original_cover


def test_hand_ownership_bridges_only_recent_direct_evidence_for_either_hand():
    from vision.hand_ownership import HandOwnership
    for mirror in [False,True]:
        h=palm();o=face()
        if mirror:
            h=h.copy();h[:,0]=640-h[:,0]
            o=face(420)
        o.wrists=np.array([h[0]])
        tracker=HandOwnership()
        assert not tracker.assign([h],[o],0) and len(o.hands)==1
        for stamp in [100,240]:
            current=face(420 if mirror else 120)
            assert not tracker.assign([h],[current],stamp) and len(current.hands)==1
        stale=face(420 if mirror else 120)
        assert tracker.assign([h],[stale],251) and not stale.hands
        # No owner is ever learned from an unassigned new hand.
        assert HandOwnership().assign([h],[stale],0)

def test_hand_ownership_rejects_crossing_and_new_hand_jump():
    from vision.hand_ownership import HandOwnership
    tracker=HandOwnership();h=palm();o=face();o.wrists=np.array([h[0]])
    tracker.assign([h],[o],0)
    a,b=face(115),face(125,1)
    assert tracker.assign([h],[a,b],100) and not a.hands and not b.hands
    moved=h+np.array([100,0])
    assert tracker.assign([moved],[face()],110)
