"""Private deployment checkpoint. Never infer privacy flags from display text."""
import json
from pathlib import Path
import httpx

def make_snapshot(state):
    if not state.get('session_id') or state.get('status')=='IDLE':raise ValueError('No active privacy session')
    if state.get('recording'):raise ValueError('Finalize the active recording before deployment')
    pending=state.get('unresolved_privacy_request',True)
    if type(pending) is not bool:raise ValueError('Invalid explicit pending-privacy flag')
    return {'session_id':state['session_id'],'timestamp_ms':state.get('timestamp_ms',0),'source':state.get('source','webcam'),
        'capture_sequence_floor':max(state.get('frame_id') or 0,state.get('metrics',{}).get('capture_frames') or 0),
        'dropped_before_admission':state.get('metrics',{}).get('dropped_before_admission',0),
        'clock_origin_monotonic':state.get('clock_origin_monotonic'),'tracks':state.get('tracks',[]),
        'unresolved_privacy_request':pending}

def main():
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=5,trust_env=False) as client:
        response=client.get('/api/state'); response.raise_for_status()
    snapshot=make_snapshot(response.json())
    path=Path(__file__).resolve().parents[1]/'.local/deployment-recovery.json'
    path.parent.mkdir(mode=0o700,exist_ok=True)
    path.write_text(json.dumps(snapshot)); path.chmod(0o600)
    print('Private checkpoint saved; pending privacy request:',snapshot['unresolved_privacy_request'])
    print('Older backends without an explicit pending flag are retained conservatively as pending.')

if __name__=='__main__':main()
