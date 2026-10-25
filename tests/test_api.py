from fastapi.testclient import TestClient
from backend.main import app
import backend.main as main
from backend.runtime import Runtime

def test_idle_api_no_camera_on_reconnect_and_local_only(monkeypatch,tmp_path):
    runtime=Runtime(artifact_root=tmp_path)
    monkeypatch.setattr(main,"runtime",runtime)
    with TestClient(app) as client:
        for _ in range(3):
            response=client.get("/api/state")
            assert response.status_code==200 and response.json()["status"]=="IDLE"
            with client.websocket_connect("/api/events") as socket:
                assert socket.receive_json()["session_id"] is None
        assert runtime.camera is None and runtime.worker is None
        assert client.post("/api/session/start",headers={"Origin":"https://unrelated.example"}).status_code==403
        assert client.post("/api/recording/start").status_code==409
        assert client.post("/api/unblur").status_code in {404,405}
        assert client.post("/api/test/inject").status_code in {404,405}
        assert client.get("/api/recordings/missing/original").status_code==404
        assert client.get("/api/recordings/missing/anything").status_code==404
        assert runtime.camera is None


def test_paired_file_availability_and_distinct_download_names(monkeypatch,tmp_path):
    import numpy as np
    from backend.recording import PairedRecorder
    runtime=Runtime(artifact_root=tmp_path)
    monkeypatch.setattr(main,'runtime',runtime)
    rec=PairedRecorder(tmp_path,64,48,'s_pair_test')
    raw=np.random.default_rng(1).integers(0,255,(48,64,3),dtype=np.uint8)
    rec.write(raw,np.zeros_like(raw),1000);rec.write(raw,np.zeros_like(raw),1050)
    meta=rec.close();runtime.recordings[rec.recording_id]=(rec.directory,meta)
    with TestClient(app) as client:
        base=f'/api/recordings/{rec.recording_id}'
        for kind in ['original','protected']:
            head=client.head(f'{base}/{kind}')
            assert head.status_code==200 and not head.content
            assert f's_pair_test_{rec.recording_id}_{kind}.mp4' in head.headers['content-disposition']
        assert client.get(base+'/original').content!=client.get(base+'/protected').content
        (rec.directory/'original.mp4').unlink()
        assert client.head(base+'/original').status_code==404
        assert client.head(base+'/protected').status_code==200
