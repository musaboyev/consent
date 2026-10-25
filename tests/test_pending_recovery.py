"""Pending requests must not deadlock reliable identity enrollment/reveal."""
import numpy as np
from test_privacy_pipeline import Detector,face,enroll,packet,request,feature
from test_reversible import peace
from vision.engine import VisionEngine


def test_pending_request_can_enroll_protected_and_peace_release_only_its_owner():
    d=Detector();e=VisionEngine('s_pending',d)
    e.mark_unresolved()
    r=enroll(e,d,[face(),face(450,1)])
    visible=[p for p in e.registry.people.values() if p.visible]
    assert len(visible)==2, 'Pending masking must not block reliable enrollment forever'
    a,b=visible
    assert a.protected and b.protected and e.unresolved
    assert not np.array_equal(r.protected_bgr,packet(600).bgr)
    v=face();v.hands=[peace()];d.observations=[v,face(450,1)]
    for t in range(700,1400,100):r=e.step(packet(t),[])
    assert not a.protected and b.protected
    assert e.unresolved and e.unresolved_exemptions=={a.person_id}
    assert np.array_equal(r.protected_bgr[140:220,140:200],packet(1300).bgr[140:220,140:200])
    assert not np.array_equal(r.protected_bgr[140:220,470:530],packet(1300).bgr[140:220,470:530])


def test_pending_recovery_preserves_absent_registered_protected_identity():
    d=Detector();e=VisionEngine('s_pending',d)
    old=enroll(e,d,[face(450,1)]).tracks[0].person_id
    e.step(packet(700),[request(e,old,700)])
    gallery=[x.copy() for x in e.registry.people[old].gallery]
    d.observations=[]
    for t in range(800,2100,100):e.step(packet(t),[])
    e.mark_unresolved()
    r=enroll(e,d,[face()],start=2200)
    newcomer=[p for p in e.registry.people.values() if p.visible]
    assert len(newcomer)==1, 'Missing protected identity must not prevent protected enrollment during a pending request'
    current=newcomer[0]
    assert current.person_id!=old and current.protected
    v=face();v.hands=[peace()];d.observations=[v]
    for t in range(2900,3600,100):e.step(packet(t),[])
    assert not current.protected and e.registry.people[old].protected
    assert all(np.array_equal(a,b) for a,b in zip(gallery,e.registry.people[old].gallery))
    # A transport gap is represented by no admitted observations; identities
    # and pending state stay in the same engine, just as PhoneCamera reconnect.
    d.observations=[]
    for t in range(3600,14000,100):
        r=e.step(packet(t),[])
        assert np.array_equal(r.protected_bgr,packet(t).bgr)
    d.observations=[face(),face(450,1)]
    for t in range(14100,14700,100):r=e.step(packet(t),[])
    assert e.registry.people[current.person_id].visible and not e.registry.people[current.person_id].protected
    assert e.registry.people[old].visible and e.registry.people[old].protected
    assert len(e.registry.people)==2


def test_pending_enrollment_rejects_ambiguous_or_occupied_identity():
    d=Detector();e=VisionEngine('s',d);pid=enroll(e,d,[face()]).tracks[0].person_id
    e.mark_unresolved()
    # Two spatially separated observations with the same face feature cannot
    # acquire two identities, even when the existing ID is already occupied.
    d.observations=[face(),face(450)]
    for t in range(700,1800,100):e.step(packet(t),[])
    assert list(e.registry.people)==[pid]
    # A possible but subthreshold match must not be relabeled as a newcomer.
    d.observations=[];e.step(packet(3000),[])
    possible=face();possible.feature=.4*feature(0)+np.sqrt(.84)*feature(1)
    d.observations=[possible]
    for t in range(3100,4400,100):e.step(packet(t),[])
    assert list(e.registry.people)==[pid]


def test_uncertain_owner_cannot_release_pending_enrollment():
    d=Detector();e=VisionEngine('s',d);e.mark_unresolved()
    pid=enroll(e,d,[face()]).tracks[0].person_id
    v=face();v.hands=[peace()];v.ownership_uncertain=True;d.observations=[v]
    for t in range(700,2000,100):e.step(packet(t),[])
    assert e.registry.people[pid].protected and not e.unresolved_exemptions


def test_failed_model_output_and_unrelated_speech_do_not_create_requests():
    from backend.meta.intent import uncertain,should_queue_privacy,Intent
    for text in ['blur me','the camera is working','请取消我脸上的模糊','hello']:
        assert not should_queue_privacy(uncertain(),text,validated=False)
    no_action=Intent(intent='NO_ACTION',target_kind='UNKNOWN',target_description=None,needs_clarification=False)
    assert not should_queue_privacy(no_action,'the camera is working',validated=True)
    assert should_queue_privacy(uncertain(),'blur that person',validated=True)
