# Pretrained model provenance

Run `.venv/bin/python scripts/download_models.py`. Exact binary sizes, upstream URLs, and SHA-256 hashes are recorded in [manifest.json](manifest.json). Reruns verify those hashes. Files below the expected binary size are rejected to catch Git LFS pointers. TLS validation remains enabled.

| File | Upstream | License |
|---|---|---|
| face_detection_yunet_2023mar.onnx | OpenCV Zoo / face_detection_yunet | MIT |
| face_recognition_sface_2021dec.onnx | OpenCV Zoo / face_recognition_sface | Apache-2.0 |
| hand_landmarker.task | Google MediaPipe model storage, float16 version 1 | Apache-2.0 |
| pose_landmarker_lite.task | Google MediaPipe model storage, float16 version 1 | Apache-2.0 |

Sources: [OpenCV face tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html), [OpenCV Zoo](https://github.com/opencv/opencv_zoo), [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).

No models are trained here. Large model binaries are ignored by source control. OpenCV sources currently use the upstream main path; the recorded hash pins the actual downloaded content and rejects future changes.
