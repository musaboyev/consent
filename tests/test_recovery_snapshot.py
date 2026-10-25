import pytest
from scripts.save_session_recovery import make_snapshot

def test_pending_privacy_is_never_reconstructed_from_a_message():
    state={'session_id':'s_real','timestamp_ms':10,'status':'RUNNING','tracks':[]}
    for reason in [None,'Privacy request pending — scene temporarily obscured','Camera unavailable','Targeted output resumed']:
        assert make_snapshot({**state,'reason':reason})['unresolved_privacy_request'] is True
        assert make_snapshot({**state,'reason':reason,'unresolved_privacy_request':True})['unresolved_privacy_request'] is True
    assert make_snapshot({**state,'unresolved_privacy_request':False})['unresolved_privacy_request'] is False
    with pytest.raises(ValueError):make_snapshot({**state,'unresolved_privacy_request':'false'})

def test_snapshot_preserves_source_clock_and_capture_sequence():
    snapshot=make_snapshot({'session_id':'s_phone','timestamp_ms':500,'status':'RUNNING','source':'iphone',
        'frame_id':30,'metrics':{'capture_frames':35,'dropped_before_admission':5},
        'clock_origin_monotonic':1234.5,'unresolved_privacy_request':True})
    assert snapshot['source']=='iphone' and snapshot['capture_sequence_floor']==35
    assert snapshot['clock_origin_monotonic']==1234.5 and snapshot['dropped_before_admission']==5
