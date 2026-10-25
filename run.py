"""Single demo command, with permission preflight before the worker starts."""
from pathlib import Path
import os
from dotenv import load_dotenv

if __name__=="__main__":
    root=Path(__file__).resolve().parent
    load_dotenv(root/".env")
    if not (root/"frontend"/"dist"/"index.html").exists():
        raise SystemExit("Build the dashboard first: cd frontend && npm ci && npm run build")
    if os.getenv("CONSENTLENS_CAMERA_SOURCE","webcam")=="webcam":
        from scripts.authorize_camera import authorize
        if not authorize():
            print("Dashboard will still start; fix Camera permission and retry Start camera session.")
    # The only concurrent capture owner is backend.capture.Camera. Preflight closed.
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"]="1"
    import uvicorn
    uvicorn.run("backend.main:app",host=os.getenv("CONSENTLENS_BIND","127.0.0.1"),port=8000,workers=1,
        timeout_graceful_shutdown=5,ws_max_size=700000)
