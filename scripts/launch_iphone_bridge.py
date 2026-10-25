"""Launch our installed bridge with local upload config; never reads Model API keys."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import httpx

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',required=True)
    parser.add_argument('--backend-url',required=True)
    args=parser.parse_args()
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=5,trust_env=False) as client:
        state=client.get('/api/state').json()
        if state.get('source')!='iphone' or state.get('status') not in {'WAITING_SOURCE','RUNNING'}:
            raise SystemExit('Start the existing backend’s iPhone session first; no session was reset')
        response=client.get('/api/source/iphone/pair');response.raise_for_status()
        token=response.json()['upload_token']
    # devicectl officially forwards DEVICECTL_CHILD_ variables. No credential in
    # command arguments, URL, saved launch configuration, stdout or reports.
    env={k:v for k,v in os.environ.items() if not k.startswith('DEVICECTL_CHILD_') and k not in {'MODEL_API_KEY','META_API_KEY','LLAMA_API_KEY'}}
    env['DEVICECTL_CHILD_CONSENTLENS_BACKEND_URL']=args.backend_url
    env['DEVICECTL_CHILD_CONSENTLENS_UPLOAD_TOKEN']=token
    env['DEVICECTL_CHILD_CONSENTLENS_AUTO_CONNECT']='1'
    output=Path(__file__).resolve().parents[1]/'.local/iphone-launch-result.json'
    result=subprocess.run(['xcrun','devicectl','device','process','launch','--device',args.device,
        '--terminate-existing','--timeout','30','--json-output',str(output),'org.consentlens.bridge'],
        env=env,capture_output=True,text=True,timeout=40)
    for text in [result.stdout,result.stderr]:
        if text:print(text.replace(token,'[redacted upload token]'))
    if output.exists():
        content=output.read_text().replace(token,'[redacted upload token]')
        output.write_text(content);output.chmod(0o600)
    if result.returncode:raise SystemExit(result.returncode)
    print('Bridge launched with local upload configuration. Allow phone Camera/Local Network prompts.')

if __name__=='__main__':main()
