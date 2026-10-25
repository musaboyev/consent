"""Project-local $20 safeguard. SQLite reservations are atomic across processes.

Charges are estimates from published rates, never an account-wide billing cap.
No credentials, audio, prompts, embeddings, or transcripts enter this ledger.
"""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
PRICING=json.loads(Path(__file__).with_name("pricing.json").read_text())
NANO=1_000_000_000
PROJECT_LIMIT_USD=Decimal("20.00")
# Spark counts private reasoning plus visible JSON against this bounded total.
MAX_OUTPUT_TOKENS=1024
MAX_AUDIO_SECONDS=300
MAX_REQUEST_SECONDS=15

class BudgetError(RuntimeError):
    pass

def nanodollars(usd):
    return int((Decimal(str(usd))*NANO).to_integral_value(rounding=ROUND_CEILING))

def text_cost(model,input_tokens,output_tokens,cached_tokens=0):
    prices=PRICING["text_per_million_tokens"].get(model)
    if prices is None:
        raise BudgetError("Meta disabled: model has no approved text price in this application")
    values=(input_tokens,output_tokens,cached_tokens)
    if any(type(n) is not int or n<0 for n in values) or cached_tokens>input_tokens:
        raise BudgetError("Meta returned invalid token usage; retaining conservative reservation")
    total=(Decimal(input_tokens-cached_tokens)*Decimal(prices["input"])+Decimal(cached_tokens)*Decimal(prices["cached_input"])+Decimal(output_tokens)*Decimal(prices["output"]))/1_000_000
    return nanodollars(total)

def reserve_text_cost(model,payload):
    output=payload.get("max_completion_tokens")
    if type(output) is not int or not 1<=output<=MAX_OUTPUT_TOKENS:
        raise BudgetError("Meta request lacks an allowed output-token bound")
    if payload.get("tools") or payload.get("stream") or any(k in payload for k in ["audio","modalities"]):
        raise BudgetError("Unmetered Meta request type disabled")
    # Text-only JSON bytes + framing allowance: deliberately conservative,
    # not a claim to reproduce Meta's tokenizer or server-side formatting.
    input_bound=len(json.dumps(payload,ensure_ascii=False).encode("utf-8"))+2048
    if input_bound>16_384:
        raise BudgetError("Meta request exceeds the application's input bound")
    return text_cost(model,input_bound,output),input_bound

def usage_cost(model,usage):
    if not isinstance(usage,dict):
        raise BudgetError("Meta response omitted token usage")
    prompt=usage.get("prompt_tokens",usage.get("input_tokens"))
    completion=usage.get("completion_tokens",usage.get("output_tokens"))
    detail=usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached=detail.get("cached_tokens",usage.get("cached_tokens",0))
    # completion_tokens includes reasoning tokens; don't add them a second time.
    return text_cost(model,prompt,completion,cached)

def audio_cost(model,seconds):
    price=PRICING["audio_per_hour"].get(model)
    if price is None:
        raise BudgetError("Meta disabled: model has no approved audio price")
    if not math.isfinite(seconds) or seconds<0:
        raise BudgetError("Invalid audio duration")
    return nanodollars(Decimal(str(seconds))*Decimal(price)/3600)

class Budget:
    def __init__(self,path=None,limit_usd=PROJECT_LIMIT_USD):
        self.path=Path(path or ROOT/".local"/"meta-budget.sqlite3")
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        limit=nanodollars(limit_usd)
        if limit<0 or limit>nanodollars(PROJECT_LIMIT_USD):
            raise BudgetError("Project Meta limit cannot exceed $20")
        try:
            with self._db() as db:
                db.execute("CREATE TABLE IF NOT EXISTS settings(id INTEGER PRIMARY KEY CHECK(id=1), limit_nano INTEGER NOT NULL, disabled_reason TEXT)")
                db.execute("CREATE TABLE IF NOT EXISTS entries(id TEXT PRIMARY KEY, model TEXT NOT NULL, reserved_nano INTEGER NOT NULL CHECK(reserved_nano>=0), charged_nano INTEGER, state TEXT NOT NULL, details TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
                db.execute("INSERT OR IGNORE INTO settings(id,limit_nano) VALUES(1,?)",(limit,))
                # Reopening cannot raise the persisted limit, including after key changes.
                db.execute("UPDATE settings SET limit_nano=MIN(limit_nano,?) WHERE id=1",(limit,))
            self.path.chmod(0o600)
        except (OSError,sqlite3.Error) as exc:
            raise BudgetError("Meta disabled: persistent budget ledger unavailable") from exc

    @contextmanager
    def _db(self):
        db=None
        try:
            db=sqlite3.connect(self.path,timeout=5,isolation_level=None)
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except sqlite3.Error as exc:
            if db is not None:db.rollback()
            raise BudgetError("Meta disabled: persistent budget ledger unavailable") from exc
        except BaseException:
            if db is not None:db.rollback()
            raise
        finally:
            if db is not None:db.close()

    @staticmethod
    def _totals(db):
        spent,reserved,uncertain=db.execute("SELECT COALESCE(SUM(COALESCE(charged_nano,0)),0), COALESCE(SUM(CASE WHEN state='RESERVED' THEN reserved_nano ELSE 0 END),0), COALESCE(SUM(CASE WHEN state='UNCERTAIN' THEN charged_nano ELSE 0 END),0) FROM entries").fetchone()
        limit,reason=db.execute("SELECT limit_nano,disabled_reason FROM settings WHERE id=1").fetchone()
        return limit,spent,reserved,uncertain,reason

    def reserve(self,model,amount,details=None,reservation_id=None):
        if type(amount) is not int or amount<0:
            raise BudgetError("Invalid Meta cost reservation")
        rid=reservation_id or "meta_"+uuid.uuid4().hex
        with self._db() as db:
            limit,spent,reserved,uncertain,reason=self._totals(db)
            if reason or spent+reserved+amount>limit:
                raise BudgetError(reason or "Meta disabled for this request: insufficient estimated project budget")
            now=time.time()
            # Store only caller-controlled accounting metadata, never request bodies.
            db.execute("INSERT INTO entries VALUES(?,?,?,?,?,?,?,?)",(rid,model,amount,None,"RESERVED",json.dumps({"pricing_checked_at":PRICING["checked_at"],**(details or {})}),now,now))
        return rid

    def settle(self,rid,amount=None,usage=None,uncertain=False):
        with self._db() as db:
            row=db.execute("SELECT reserved_nano,state FROM entries WHERE id=?",(rid,)).fetchone()
            if row is None:raise BudgetError("Unknown Meta reservation")
            reserved,state=row
            if state!="RESERVED":return  # one settlement only
            if amount is None:
                amount=reserved; uncertain=True
            if type(amount) is not int or amount<0:raise BudgetError("Invalid Meta reconciliation")
            details=json.loads(db.execute("SELECT details FROM entries WHERE id=?",(rid,)).fetchone()[0])
            if usage is not None:details["returned_usage"]=usage
            db.execute("UPDATE entries SET charged_nano=?,state=?,details=?,updated_at=? WHERE id=?",(amount,"UNCERTAIN" if uncertain else "RECONCILED",json.dumps(details),time.time(),rid))
            if amount>reserved:
                db.execute("UPDATE settings SET disabled_reason=? WHERE id=1",("Meta disabled: returned usage exceeded reservation; cost estimate requires review",))

    def status(self):
        with self._db() as db:
            limit,spent,reserved,uncertain,reason=self._totals(db)
            count=db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            held=db.execute("SELECT COUNT(*) FROM entries WHERE state='RESERVED'").fetchone()[0]
        remaining=max(0,limit-spent-reserved)
        disabled=bool(reason) or remaining<=0
        return {"limit_usd":limit/NANO,"estimated_consumed_usd":spent/NANO,"reserved_usd":reserved/NANO,
            "remaining_estimated_usd":remaining/NANO,"uncertain_estimate_usd":uncertain/NANO,
            "requests_accounted":count,"outstanding_reservations":held,"meta_disabled":disabled,
            "reason":reason or ("Meta budget exhausted or fully reserved; local vision continues" if disabled else None),
            "pricing_checked_at":PRICING["checked_at"],"pricing_source":PRICING["source"],
            "scope":"ConsentLens Meta calls only; local safeguard, not account-wide billing cap"}

    def assert_available(self):
        status=self.status()
        if status["meta_disabled"]:
            raise BudgetError(status["reason"])

    def account_legacy_probe(self):
        # One real Spark request completed before the user added this safeguard.
        # Returned usage was not retained. Count a labeled conservative estimate.
        proof=ROOT/"evaluation"/"meta-access-pre-budget.json"
        if not proof.exists():return
        rid="pre_budget_spark_probe_20261024"
        with self._db() as db:
            if db.execute("SELECT 1 FROM entries WHERE id=?",(rid,)).fetchone():return
            amount=nanodollars("0.02")
            now=time.time()
            details={"note":"One pre-budget Spark probe; usage not retained. Conservative $0.02 allowance; actual invoice charge unknown."}
            db.execute("INSERT INTO entries VALUES(?,?,?,?,?,?,?,?)",(rid,"muse-spark-1.3",amount,amount,"UNCERTAIN",json.dumps(details),now,now))

@lru_cache(maxsize=1)
def get_budget():
    ledger=Budget()
    ledger.account_legacy_probe()
    return ledger

def public_budget_status():
    try:
        return get_budget().status()
    except (BudgetError,OSError):
        return {"limit_usd":20,"remaining_estimated_usd":None,"meta_disabled":True,"reason":"Meta disabled: persistent budget ledger unavailable; local vision continues"}

class AudioLease:
    """Reserve a bounded stream before connect; track attempted bytes before send."""
    def __init__(self,budget=None,model=None,max_seconds=MAX_AUDIO_SECONDS):
        self.budget=budget or get_budget()
        self.model=model or os.getenv("META_ASR_MODEL","muse-voice-transcribe-1.0")
        self.max_seconds=min(float(max_seconds),MAX_AUDIO_SECONDS)
        if self.max_seconds<=0:raise BudgetError("Invalid audio stream limit")
        self.rid=self.budget.reserve(self.model,audio_cost(self.model,self.max_seconds),{"kind":"audio","max_seconds":self.max_seconds})
        self.attempted_bytes=0; self.progress_ms=0; self.started=None; self.closed=False

    def check(self):
        self.budget.assert_available()
        if self.started is not None and time.monotonic()-self.started>=self.max_seconds:
            raise BudgetError("Meta microphone stopped: bounded stream duration reached; restart voice to continue")

    def before_send(self,chunk):
        self.check()
        if self.started is None:self.started=time.monotonic()
        if len(chunk)%2 or self.attempted_bytes+len(chunk)>self.max_seconds*48000:
            raise BudgetError("Meta microphone stopped: audio duration limit reached")
        self.attempted_bytes+=len(chunk)

    def observe(self,event):
        offset=event.get("audioProcessedMs")
        if isinstance(offset,(int,float)) and math.isfinite(offset) and offset>=0:
            self.progress_ms=max(self.progress_ms,offset)

    def close(self,clean=False):
        if self.closed:return
        sent_seconds=self.attempted_bytes/48000
        # Meta meters whole processed seconds. Sent audio is a conservative upper
        # estimate when progress lags; processed offsets aren't invoice usage.
        seconds=math.floor(max(sent_seconds,self.progress_ms/1000))
        charge=audio_cost(self.model,seconds)
        self.budget.settle(self.rid,charge,{"audio_attempted_seconds":sent_seconds,"audio_progress_ms":self.progress_ms,"metered_seconds_estimate":seconds},uncertain=not clean or self.progress_ms<self.attempted_bytes/48)
        self.closed=True
