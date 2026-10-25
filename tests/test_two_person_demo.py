"""Controlled observations for the exact A/B demo; not physical detector evidence."""
import numpy as np
from test_privacy_pipeline import Detector, face, palm, packet, enroll
from test_reversible import peace
from vision.engine import VisionEngine


def test_each_person_independently_protects_leaves_returns_and_reveals():
    detector=Detector(); engine=VisionEngine('s_two_person_controlled',detector)
    initial=enroll(engine,detector,[face(120,0),face(400,1)])
    ids=[t.person_id for t in initial.tracks]
    assert len(ids)==2 and len(set(ids))==2
    now=700;events=[]
    def tick(observations):
        nonlocal now
        detector.observations=observations;source=packet(now)
        before=source.bgr.copy();result=engine.step(source,[])
        assert np.array_equal(source.bgr,before)
        assert result.protection_mode not in {'SCENE_SHIELD','ERROR_SHIELD'}
        assert np.array_equal(result.protected_bgr[:40],source.bgr[:40])
        events.extend(result.events);now+=100
        return result,source
    for actor in [0,1]:
        neighbor=1-actor;positions=[120,400];x=positions[actor];other_x=positions[neighbor]
        for _ in range(13):
            observations=[face(120,0),face(400,1)]
            observations[actor].hands=[palm(x)]
            observations[actor].quality=False;observations[actor].feature=None
            result,source=tick(observations)
            assert np.array_equal(result.protected_bgr[140:220,other_x+20:other_x+80],source.bgr[140:220,other_x+20:other_x+80])
        assert engine.registry.people[ids[actor]].protected
        assert not engine.registry.people[ids[neighbor]].protected
        for _ in range(6):
            result,source=tick([face(120,0),face(400,1)])
            assert not np.array_equal(result.protected_bgr[140:220,x+20:x+80],source.bgr[140:220,x+20:x+80])
        # Exact ten seconds: neighbor remains visible and byte-identical.
        for _ in range(101):
            result,source=tick([face(other_x,neighbor)])
            assert np.array_equal(result.protected_bgr,source.bgr)
            assert engine.registry.people[ids[actor]].protected
            assert not engine.registry.people[ids[actor]].visible
        for _ in range(8):
            result,source=tick([face(120,0),face(400,1)])
            assert not np.array_equal(result.protected_bgr[140:220,x+20:x+80],source.bgr[140:220,x+20:x+80])
            assert np.array_equal(result.protected_bgr[140:220,other_x+20:other_x+80],source.bgr[140:220,other_x+20:other_x+80])
        assert engine.registry.people[ids[actor]].visible
        assert len(engine.registry.people)==2
        for _ in range(12):
            observations=[face(120,0),face(400,1)]
            observations[actor].hands=[peace()+np.array([x-120,0])]
            result,source=tick(observations)
        assert not engine.registry.people[ids[actor]].protected
        assert not engine.registry.people[ids[neighbor]].protected
        assert np.array_equal(result.protected_bgr,source.bgr)
        for _ in range(5):tick([face(120,0),face(400,1)])
    for pid in ids:
        assert sum(e.type=='PROTECTION_ACTIVATED' and e.person_id==pid for e in events)==1
        assert sum(e.type=='PROTECTION_REMOVED' and e.person_id==pid for e in events)==1
        assert any(e.type=='PERSON_REIDENTIFIED' and e.person_id==pid and e.details['restored_privacy']=='BLURRED' for e in events)


def test_revealing_a_does_not_reveal_protected_b():
    d=Detector();e=VisionEngine('s_two_protected',d)
    ids=[t.person_id for t in enroll(e,d,[face(120,0),face(400,1)]).tracks]
    for t in range(700,2000,100):
        obs=[face(120,0),face(400,1)]
        for x,o in zip([120,400],obs):o.hands=[palm(x)];o.quality=False;o.feature=None
        d.observations=obs;e.step(packet(t),[])
    assert all(e.registry.people[pid].protected for pid in ids)
    d.observations=[face(120,0),face(400,1)]
    for t in range(2000,2500,100):e.step(packet(t),[])
    a=face();a.hands=[peace()];d.observations=[a,face(400,1)]
    for t in range(2500,3700,100):result=e.step(packet(t),[])
    assert not e.registry.people[ids[0]].protected
    assert e.registry.people[ids[1]].protected
    assert np.array_equal(result.protected_bgr[140:220,140:200],packet(3600).bgr[140:220,140:200])
    assert not np.array_equal(result.protected_bgr[140:220,420:480],packet(3600).bgr[140:220,420:480])
