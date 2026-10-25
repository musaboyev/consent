"""Opt-in real Spark TEXT checks; no microphone, registry actions or fake ASR."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
from backend.meta.intent import Spark
from backend.meta.budget import public_budget_status

async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real',action='store_true',help='Make budgeted Meta requests')
    if not parser.parse_args().real:
        parser.error('--real is required; this consumes the existing project Meta budget')
    load_dotenv(ROOT/'.env')
    if not os.getenv('MODEL_API_KEY'):
        raise SystemExit('MODEL_API_KEY missing; no requests made')
    cases=json.loads((ROOT/'evaluation/multilingual-intent-cases.json').read_text())
    report={'evidence':'Real Meta Spark text-only requests; NOT spoken transcription or live actions',
            'model':os.getenv('META_SPARK_MODEL','muse-spark-1.3'),
            'budget_before':public_budget_status(),'results':[]}
    for case in cases:
        spark=Spark(); started=time.monotonic()
        intent,route=await spark.classify(case['text'],force=True)
        passed=spark.last_status=='RESPONDED' and intent.intent in case['expected']
        if case.get('clarification_required'):
            passed=passed and intent.needs_clarification and intent.target_kind!='SELF'
        elif intent.intent in {'PROTECT_PERSON','UNPROTECT_PERSON'}:
            passed=passed and intent.target_kind=='SELF' and not intent.needs_clarification
        row={**case,'passed':passed,'result':intent.model_dump(),'status':spark.last_status,
             'error':spark.last_error,'usage':spark.last_usage,'elapsed_ms':round((time.monotonic()-started)*1000)}
        report['results'].append(row)
        report['budget_after']=public_budget_status()
        (ROOT/'evaluation/multilingual-real-spark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'language':case['language'],'text':case['text'],'passed':passed,'intent':intent.intent,'status':spark.last_status},ensure_ascii=False),flush=True)
        if spark.last_status in {'BUDGET_DISABLED','FAILED','BLOCKED'}:
            print('Stopped after provider/budget failure; no automatic retries.',flush=True)
            break
    raise SystemExit(0 if len(report['results'])==len(cases) and all(r['passed'] for r in report['results']) else 1)

if __name__=='__main__':
    asyncio.run(main())
