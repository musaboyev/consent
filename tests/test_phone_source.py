"""Transport fixtures test integration policy, not actual iPhone SDK delivery."""
import time
import uuid
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.phone_source import PhoneCamera
from backend.runtime import Runtime
from vision.engine import VisionEngine
from tests.test_privacy_pipeline import Detector,face,request,wait_until,decode

def test_native_websocket_upload_enters_existing_runtime(monkeypatch,tmp_path):
    import backend.main as main
    detector=Detector(); detector.observations=[face()]
    runtime=Runtime(lambda sid:VisionEngine(sid,detector),artifact_root=tmp_path)
    monkeypatch.setattr(main,'runtime',runtime)
    with TestClient(main.app) as client:
        token=client.get('/api/source/iphone/pair').json()['upload_token']
        client.post('/api/session/start?source=iphone')
        wait_until(lambda:isinstance(runtime.camera,PhoneCamera))
        wait_until(lambda:runtime.state()['status']=='WAITING_SOURCE')
        assert runtime.state()['source']=='iphone'
        sid=runtime.session_id; camera=runtime.camera
        with client.websocket_connect('/api/source/iphone/frames') as socket:
            socket.send_json({'upload_token':token})
            assert socket.receive_json()['session_id']==sid
            jpeg,metadata=payload()
            socket.send_json(metadata); socket.send_bytes(jpeg)
            assert socket.receive_json()['admitted_frame_id']==1
            wait_until(lambda:runtime.state().get('frame_id')==1)
            assert runtime.state()['source_metadata']['source']=='meta_dat_mock_live_iphone'
        assert runtime.camera is camera and runtime.session_id==sid
        assert client.get('/api/state',headers={'host':'192.168.1.20:8000'}).status_code==403

def payload(value=100,sequence=1):
    image=np.full((640,360,3),value,np.uint8)
    cv2.rectangle(image,(80,200),(280,400),(0,220,0),-1)
    ok,jpeg=cv2.imencode('.jpg',image); assert ok
    return jpeg.tobytes(),{'sequence':sequence,'sdk_timestamp_ms':sequence*70,'orientation':'up','provenance':'meta_dat_mock_live_iphone'}

def test_phone_bounded_mailbox_orientation_and_connection_ownership():
    camera=PhoneCamera(time.monotonic()); camera.start(); camera.attach('one')
    try:
        for index in range(1,4):camera.admit(*payload(sequence=index),'one')
        assert camera.frames.qsize()==1 and camera.dropped==2
        packet=camera.frames.get_nowait()
        assert packet.bgr.shape==(480,640,3) and not packet.bgr.flags.writeable
        assert not packet.bgr[:,0:150].any() # portrait letterbox, no stretched face
        assert packet.source_metadata['source_sequence']==3
        with pytest.raises(ValueError):camera.admit(*payload(sequence=2),'one')
        with pytest.raises(ValueError):camera.attach('two')
        camera.detach('one'); camera.attach('two')
        camera.admit(*payload(sequence=1),'two')
        camera.detach('one'); assert camera.connected # stale close cannot detach new source
    finally:camera.close()

def test_phone_disconnect_reconnect_same_registry_and_paired_recording(tmp_path):
    detector=Detector(); detector.observations=[face()]
    runtime=Runtime(lambda sid:VisionEngine(sid,detector),artifact_root=tmp_path)
    runtime.start(source='iphone')
    try:
        wait_until(lambda:isinstance(runtime.camera,PhoneCamera))
        camera=runtime.camera; camera.attach('one')
        for seq in range(1,10):
            camera.admit(*payload(sequence=seq),'one'); time.sleep(.10)
        wait_until(lambda:any(t['identity_status']=='KNOWN' for t in runtime.state()['tracks']))
        sid=runtime.session_id; pid=runtime.state()['tracks'][0]['person_id']
        runtime.command('recording_start'); runtime.enqueue(request(runtime.engine,pid,runtime.now_ms()))
        camera.admit(*payload(sequence=10),'one')
        wait_until(lambda:runtime.engine.registry.people[pid].protected)
        camera.detach('one')
        wait_until(lambda:runtime.state()['protection_mode']=='ERROR_SHIELD')
        assert runtime.engine.registry.people[pid].protected and runtime.session_id==sid
        camera.attach('two')
        # A live source must continue producing while identity converges; a short
        # finite burst followed by silence can correctly return to a source slate.
        deadline=time.monotonic()+8; seq=0
        while time.monotonic()<deadline:
            seq+=1; camera.admit(*payload(value=160,sequence=seq),'two'); time.sleep(.10)
            state=runtime.state()
            if state['output_state']=='NORMAL' and state.get('recording',{}).get('frames',0)>=4:break
        assert runtime.state()['output_state']=='NORMAL'
        assert runtime.session_id==sid and runtime.camera is camera
        assert runtime.engine.registry.people[pid].protected
        result=runtime.command('recording_stop')
        directory=runtime.recordings[result['recording_id']][0]
        a,b=decode(directory/'original.mp4'),decode(directory/'protected.mp4')
        assert len(a)==len(b)>2 and [f[0] for f in a]==[f[0] for f in b]
    finally:runtime.end()
