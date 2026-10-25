"""Download versioned upstream binaries; verify against the recorded manifest on reruns."""
import hashlib
import json
from pathlib import Path
import urllib.request
import ssl

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "face_detection_yunet_2023mar.onnx": ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", "MIT", 200_000),
    "face_recognition_sface_2021dec.onnx": ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx", "Apache-2.0", 30_000_000),
    "hand_landmarker.task": ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", "Apache-2.0", 5_000_000),
    "pose_landmarker_lite.task": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", "Apache-2.0", 3_000_000),
}

def main():
    # python.org macOS builds may not have run Install Certificates.command yet.
    # Use a trusted CA bundle, never disable certificate validation.
    try:
        import certifi
        ca_file = certifi.where()
    except ImportError:
        ca_file = "/etc/ssl/cert.pem"
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca_file))))
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for name, (url, license_name, minimum) in MODELS.items():
        path = directory / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            temporary = path.with_suffix(path.suffix + ".download")
            urllib.request.urlretrieve(url, temporary)
            if temporary.stat().st_size < minimum:
                raise RuntimeError(f"{name}: too small; may be a Git LFS pointer")
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size < minimum or (name in previous and previous[name]["sha256"] != digest):
            raise RuntimeError(f"{name}: size/checksum mismatch")
        manifest[name] = {"url": url, "license": license_name, "bytes": path.stat().st_size, "sha256": digest}
        print(f"Verified {name}: {digest}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    main()
