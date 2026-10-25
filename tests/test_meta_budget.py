import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import ssl
import httpx
import pytest
from backend.meta.budget import Budget,BudgetError,AudioLease,nanodollars,reserve_text_cost,usage_cost,audio_cost
from backend.meta.intent import Spark
import backend.meta.transcribe as transcribe

def test_shared_budget_atomic_across_clients_and_restarts(tmp_path):
    path=tmp_path/"budget.sqlite3"
    Budget(path,limit_usd="0.02")
    def reserve(index):
        try:
            return Budget(path,limit_usd="0.02").reserve("muse-spark-1.3" if index%2 else "muse-voice-transcribe-1.0",nanodollars("0.002"))
        except BudgetError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted=[rid for rid in pool.map(reserve,range(24)) if rid]
    assert len(accepted)==10
    reopened=Budget(path,limit_usd="0.02")
    assert reopened.status()["remaining_estimated_usd"]==0
    assert reopened.status()["reserved_usd"]==pytest.approx(.02)
    assert reopened.status()["meta_disabled"]
    # Lost-process reservations remain held, never silently released on restart.
    reopened.settle(accepted[0],nanodollars("0.001"))
    assert Budget(path).status()["remaining_estimated_usd"]==pytest.approx(.001)
    assert Budget(path).status()["limit_usd"]==pytest.approx(.02)

def test_usage_reconciliation_cache_reasoning_and_unknown_prices(tmp_path):
    usage={"prompt_tokens":1000,"completion_tokens":200,"prompt_tokens_details":{"cached_tokens":400},"completion_tokens_details":{"reasoning_tokens":150}}
    assert usage_cost("muse-spark-1.3",usage)==nanodollars("0.00166")
    # Output already includes reasoning: do not charge 350 output tokens.
    ledger=Budget(tmp_path/"b.sqlite3")
    rid=ledger.reserve("muse-spark-1.3",nanodollars(".01"))
    ledger.settle(rid,usage_cost("muse-spark-1.3",usage),usage)
    assert Budget(ledger.path).status()["estimated_consumed_usd"]==pytest.approx(.00166)
    with pytest.raises(BudgetError):reserve_text_cost("unknown-model",{"max_completion_tokens":100})
    with pytest.raises(BudgetError):reserve_text_cost("muse-spark-1.3",{"max_completion_tokens":100,"tools":[{"type":"web_search"}]})
    with pytest.raises(BudgetError):reserve_text_cost("muse-spark-1.3",{"max_completion_tokens":1025})
    with pytest.raises(BudgetError):usage_cost("muse-spark-1.3",{"prompt_tokens":1,"completion_tokens":-1})

def test_uncertain_charge_and_overrun_disable_are_persistent(tmp_path):
    ledger=Budget(tmp_path/"b.sqlite3")
    rid=ledger.reserve("muse-spark-1.3",nanodollars(".01"))
    ledger.settle(rid)
    assert Budget(ledger.path).status()["uncertain_estimate_usd"]==pytest.approx(.01)
    rid=ledger.reserve("muse-spark-1.3",nanodollars(".001"))
    ledger.settle(rid,nanodollars(".002"))
    assert Budget(ledger.path).status()["meta_disabled"]
    with pytest.raises(BudgetError):ledger.reserve("muse-spark-1.3",1)

def test_audio_bound_and_reconcile_sent_not_just_lagging_progress(tmp_path):
    ledger=Budget(tmp_path/"b.sqlite3")
    lease=AudioLease(ledger,max_seconds=2)
    lease.before_send(bytes(48000))
    lease.observe({"audioProcessedMs":500})
    lease.close(clean=True)
    assert ledger.status()["estimated_consumed_usd"]==pytest.approx(.00005)
    assert ledger.status()["uncertain_estimate_usd"]==pytest.approx(.00005)
    lease=AudioLease(ledger,max_seconds=1)
    lease.before_send(bytes(48000))
    with pytest.raises(BudgetError):lease.before_send(bytes(2))
    lease.close()
    assert ledger.status()["reserved_usd"]==0
    assert audio_cost("muse-voice-transcribe-1.0",3600)==nanodollars(".18")

def test_exhausted_budget_prevents_any_spark_network_call(monkeypatch,tmp_path):
    ledger=Budget(tmp_path/"b.sqlite3",limit_usd=0)
    monkeypatch.setenv("MODEL_API_KEY","test-only-placeholder")
    def forbidden(*args,**kwargs):raise AssertionError("Network must not start")
    monkeypatch.setattr(httpx,"AsyncClient",forbidden)
    spark=Spark(ledger)
    intent,_=asyncio.run(spark.classify("Keep me out of this recording",force=True))
    assert intent.intent=="NEEDS_CLARIFICATION" and spark.last_status=="BUDGET_DISABLED"

def test_spark_counts_actual_returned_usage(monkeypatch,tmp_path):
    monkeypatch.setenv("MODEL_API_KEY","test-only-placeholder")
    ledger=Budget(tmp_path/"b.sqlite3")
    class Client:
        def __init__(self,*args,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,*args,**kwargs):
            assert ledger.status()["reserved_usd"]>0
            return httpx.Response(200,json={"usage":{"prompt_tokens":1000,"completion_tokens":200,"prompt_tokens_details":{"cached_tokens":400}},"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"intent":"PROTECT_PERSON","target_kind":"SELF","target_description":None,"needs_clarification":False})}}]})
    monkeypatch.setattr(httpx,"AsyncClient",Client)
    spark=Spark(ledger)
    intent,_=asyncio.run(spark.classify("Keep my face out of the recording",force=True))
    assert intent.intent=="PROTECT_PERSON" and spark.last_status=="RESPONDED"
    assert ledger.status()["estimated_consumed_usd"]==pytest.approx(.00166)
    assert ledger.status()["reserved_usd"]==0

def test_timeout_keeps_conservative_cost(monkeypatch,tmp_path):
    monkeypatch.setenv("MODEL_API_KEY","test-only-placeholder")
    ledger=Budget(tmp_path/"b.sqlite3")
    class Client:
        def __init__(self,*args,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,*args,**kwargs):raise httpx.ReadTimeout("synthetic timeout")
    monkeypatch.setattr(httpx,"AsyncClient",Client)
    asyncio.run(Spark(ledger).classify("Please leave my face out",force=True))
    status=Budget(ledger.path).status()
    assert status["estimated_consumed_usd"]>0 and status["uncertain_estimate_usd"]>0
    assert status["reserved_usd"]==0

def test_budget_exhaustion_closes_active_asr_socket(monkeypatch,tmp_path):
    monkeypatch.setenv("MODEL_API_KEY","test-only-placeholder")
    ledger=Budget(tmp_path/"b.sqlite3",limit_usd=".001")
    class Socket:
        close_code=None; closed=False
        async def send(self,payload):pass
        async def recv(self):return json.dumps({"sessionId":"test-provider"})
        async def close(self):self.closed=True; self.close_code=1000
    socket=Socket()
    async def connect(*args,**kwargs):
        assert kwargs["ssl"].verify_mode==ssl.CERT_REQUIRED
        return socket
    monkeypatch.setattr(transcribe.websockets,"connect",connect)
    async def scenario():
        with pytest.raises(BudgetError):
            async with transcribe.connection(max_seconds=2,budget=ledger) as (ws,sid):
                rid=ledger.reserve("muse-spark-1.3",nanodollars(".0009"))
                ledger.settle(rid,nanodollars(".0009"))
                await ws.send(bytes(3840))
    asyncio.run(scenario())
    assert socket.closed
    assert ledger.status()["reserved_usd"]==0

@pytest.mark.parametrize('finish,content,parser',[
    ('length','{"intent":"UNPROTECT_PERSON"','NOT_RUN'),
    ('content_filter','', 'NOT_RUN'),
    ('stop','{"intent":"UNPROTECT_PERSON"}', 'REJECTED'),
])
def test_incomplete_or_invalid_output_never_reveals_or_retries(monkeypatch,tmp_path,finish,content,parser):
    monkeypatch.setenv('MODEL_API_KEY','test-only-placeholder')
    ledger=Budget(tmp_path/'b.sqlite3');calls=[]
    class Client:
        def __init__(self,*a,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def post(self,*a,**kw):
            calls.append(kw['json'])
            assert ledger.status()['reserved_usd']>0
            return httpx.Response(200,json={'usage':{'prompt_tokens':710,'completion_tokens':1024,
                'completion_tokens_details':{'reasoning_tokens':1000}},'choices':[
                {'finish_reason':finish,'message':{'content':content}}]})
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    spark=Spark(ledger)
    result,_=asyncio.run(spark.classify('请取消我脸上的模糊',force=True))
    assert result.intent=='NEEDS_CLARIFICATION' and result.needs_clarification
    assert spark.last_status=='FAILED' and spark.last_usage['parser_status']==parser
    assert spark.last_usage['finish_reason']==finish
    assert spark.last_usage['reasoning_tokens']==1000
    assert len(calls)==1 and calls[0]['max_completion_tokens']==1024
    assert ledger.status()['reserved_usd']==0
