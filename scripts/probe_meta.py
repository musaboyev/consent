"""Small real API probes. Never print credentials, headers, or response dumps."""
import asyncio
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from backend.meta.intent import Spark,safe_error
from backend.meta.transcribe import probe
from backend.meta.budget import public_budget_status

async def main():
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/".env")
    result={"source":"actual Meta Model API probes","credential_present":bool(os.getenv("MODEL_API_KEY"))}
    if not result["credential_present"]:
        result["status"]="BLOCKED: MODEL_API_KEY missing; no request made"
    else:
        spark=Spark()
        intent,route=await spark.classify("I would rather my face did not appear in this recording.",force=True)
        result["spark"]={"status":spark.last_status,"error":spark.last_error,"intent":intent.model_dump(),"usage":spark.last_usage,"model":os.getenv("META_SPARK_MODEL","muse-spark-1.3")}
        try:
            result["transcription"]=await probe()
        except Exception as exc:
            result["transcription"]={"status":"FAILED","error":safe_error(f"{type(exc).__name__}: {exc}")}
    result["budget"]=public_budget_status()
    (root/"evaluation").mkdir(exist_ok=True)
    (root/"evaluation"/"meta-access.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    asyncio.run(main())
