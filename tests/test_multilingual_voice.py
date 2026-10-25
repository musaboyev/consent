"""Transport/routing regressions, not evidence of actual speech recognition."""
import asyncio
import json
from pathlib import Path
import httpx
import pytest
from backend.meta.budget import Budget
from backend.meta.intent import Spark, Intent, fixed_phrase, should_queue_privacy
from backend.meta.transcribe import handshake
from backend.meta.turns import Turns

CASES=json.loads((Path(__file__).parents[1]/'evaluation/multilingual-intent-cases.json').read_text())

def test_asr_autodetect_keeps_existing_audio_protocol(monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY','test-placeholder')
    config=handshake()
    assert 'languageBias' not in config
    assert config['audioEncoding']=='PCM_24KHZ'
    assert config['mode']=='ENDPOINTING'
    assert {'取消模糊','我的脸','blur','unblur','размытие','desenfoque'} <= set(config['keywords'])
    assert 'context' not in config and 'prompt' not in config

@pytest.mark.parametrize('text',['不要 unblur me','не unblur me','请 blur me'])
def test_mixed_language_cannot_be_stripped_into_english_fast_path(text):
    assert fixed_phrase(text) is None

@pytest.mark.parametrize('case',CASES)
def test_original_unicode_reaches_budgeted_spark_once(case,monkeypatch,tmp_path):
    monkeypatch.setenv('MODEL_API_KEY','test-placeholder')
    turns=Turns()
    event={'type':'speechComplete','turnId':1,'transcript':case['text'],'audioProcessedMs':1000}
    turn=turns.feed(event,0)
    assert turn.text==case['text']
    assert turns.feed(event,0) is None
    budget=Budget(tmp_path/'budget.sqlite3')
    calls=[]
    # Mock output only tests wiring; real semantic results are separately recorded.
    response_intent=Intent(intent=case['expected'][0],target_kind='DESCRIPTION' if case.get('clarification_required') else 'SELF',
                           target_description='other person' if case.get('clarification_required') else None,
                           needs_clarification=bool(case.get('clarification_required')))
    class Client:
        def __init__(self,*a,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def post(self,url,**kwargs):
            payload=kwargs['json']; calls.append(payload)
            assert payload['messages'][-1]['content']==case['text']
            assert payload['max_completion_tokens']==1024
            assert budget.status()['reserved_usd']>0
            return httpx.Response(200,json={'usage':{'prompt_tokens':100,'completion_tokens':40},'choices':[
                {'finish_reason':'stop','message':{'content':response_intent.model_dump_json()}}]})
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    result,_=asyncio.run(Spark(budget).classify(turn.text,force=True))
    assert len(calls)==1
    assert result.intent==response_intent.intent
    if result.intent in {'PROTECT_PERSON','UNPROTECT_PERSON'}:
        assert should_queue_privacy(result,turn.text)
    if case.get('clarification_required'):
        assert result.needs_clarification and result.target_kind=='DESCRIPTION'
    assert budget.status()['reserved_usd']==0
