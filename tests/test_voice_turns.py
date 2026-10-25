from backend.meta.turns import Turns
from backend.meta.intent import Intent,uncertain,should_queue_privacy

def test_partials_final_and_complete_deduplicate():
    turns=Turns()
    assert turns.feed({"type":"speechStart","turnId":1,"audioProcessedMs":100},1000) is None
    assert turns.feed({"type":"transcript","transcript":"Blur","final":False,"audioProcessedMs":400},1000) is None
    turns.feed({"type":"speechEnd","turnId":1,"audioProcessedMs":800},1000)
    result=turns.feed({"type":"speechComplete","turnId":1,"transcript":"Blur me.","audioProcessedMs":800},1000)
    assert (result.start_ms,result.end_ms,result.text)==(1100,1800,"Blur me.")
    assert turns.feed({"type":"transcript","transcript":"Blur me.","final":True,"audioProcessedMs":800},1000) is None

def test_overlapping_turns_and_stream_final_use_original_audio_window():
    turns=Turns()
    turns.feed({"type":"speechStart","turnId":1,"audioProcessedMs":100},1000)
    turns.feed({"type":"speechStart","turnId":2,"audioProcessedMs":500},1000)
    first=turns.feed({"type":"speechComplete","turnId":1,"transcript":"Hide my face","audioProcessedMs":600},1000)
    second=turns.feed({"type":"speechComplete","turnId":2,"transcript":"Please","audioProcessedMs":900},1000)
    assert first.start_ms==1100 and second.start_ms==1500
    fallback=Turns(mode="PUSH_TO_TALK").feed({"type":"transcript","transcript":"Blur me","final":True,"audioProcessedMs":3000},1000)
    assert fallback.start_ms==1000 and fallback.end_ms==4000

def test_background_ambiguity_or_provider_failure_is_not_a_privacy_objection():
    assert not should_queue_privacy(uncertain(),"What?")
    assert not should_queue_privacy(uncertain(),"Guys, where is the charger?")
    assert should_queue_privacy(uncertain(),"Please don't record my face")
    varied=Intent(intent="PROTECT_PERSON",target_kind="SELF",target_description=None,needs_clarification=False)
    assert should_queue_privacy(varied,"I would prefer to stay out of this")

def test_endpointing_final_fragment_cannot_preempt_completed_negation():
    turns=Turns()
    turns.feed({'type':'speechStart','turnId':7,'audioProcessedMs':100},1000)
    assert turns.feed({'type':'transcript','final':True,'transcript':'取消模糊','audioProcessedMs':400},1000) is None
    turns.feed({'type':'speechEnd','turnId':7,'audioProcessedMs':900},1000)
    complete={'type':'speechComplete','turnId':7,'transcript':'不要取消模糊','audioProcessedMs':1000}
    turn=turns.feed(complete,1000)
    assert turn.text=='不要取消模糊' and (turn.start_ms,turn.end_ms)==(1100,1900)
    assert turns.feed(complete,1000) is None
    assert turns.feed({'type':'transcript','final':True,'transcript':'取消模糊','audioProcessedMs':1100},1000) is None
