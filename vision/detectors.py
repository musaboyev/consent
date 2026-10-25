"""Pretrained YuNet/SFace and MediaPipe Tasks, all in source pixel coordinates."""
from dataclasses import dataclass, field
from pathlib import Path
import os
import tempfile
import cv2
import numpy as np
from vision.association import distance, HeadAnchors
from vision.identity import normalize
from vision.hand_ownership import HandOwnership
from vision.gestures import hand_covers_eye, PALM

@dataclass
class Observation:
    box: tuple[float, float, float, float]
    eyes: np.ndarray | None = None
    wrists: np.ndarray | None = None
    hands: list[np.ndarray] = field(default_factory=list)
    feature: np.ndarray | None = None
    quality: bool = False
    ownership_uncertain: bool = False
    face_detected: bool = False
    head_reliable: bool = False

class Detectors:
    def __init__(self, model_dir=None):
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "consentlens-matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "consentlens-cache"))
        import mediapipe as mp
        self.mp = mp
        directory = Path(model_dir or Path(__file__).resolve().parents[1] / "models")
        self.face = cv2.FaceDetectorYN.create(str(directory / "face_detection_yunet_2023mar.onnx"), "", (640,480), .8, .3, 500)
        self.recognizer = cv2.FaceRecognizerSF.create(str(directory / "face_recognition_sface_2021dec.onnx"), "")
        options = mp.tasks.BaseOptions
        vision = mp.tasks.vision
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=options(model_asset_path=str(directory / "pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=4,
            min_pose_detection_confidence=.45, min_pose_presence_confidence=.45,
            min_tracking_confidence=.5, output_segmentation_masks=False))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=options(model_asset_path=str(directory / "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_hands=4,
            min_hand_detection_confidence=.4, min_hand_presence_confidence=.4, min_tracking_confidence=.5))
        self.last_timestamp = -1
        self.optical=HeadAnchors()
        self.hand_ownership=HandOwnership()

    def detect(self, image, timestamp_ms):
        height, width = image.shape[:2]
        stamp = max(timestamp_ms, self.last_timestamp+1)
        self.last_timestamp = stamp
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        poses = self.pose.detect_for_video(mp_image, stamp)
        hands = self.hand.detect_for_video(mp_image, stamp)
        # Local diagnostic hooks; never included in API metadata or network calls.
        self.last_hands = [np.array([[p.x*width,p.y*height] for p in marks],dtype=np.float32) for marks in hands.hand_landmarks]
        self.last_pose_wrists = [[(round(p.x*width,1),round(p.y*height,1),round(p.visibility,2)) for p in (marks[15],marks[16])] for marks in poses.pose_landmarks]
        observations = []
        for landmarks in poses.pose_landmarks:
            xy = np.array([[p.x*width, p.y*height] for p in landmarks], dtype=np.float32)
            # Pose fallback keeps a current head/eye anchor when hands hide YuNet.
            if max(landmarks[0].visibility, landmarks[2].visibility, landmarks[5].visibility) < .4:
                continue
            eyes = xy[[2,5]]
            midpoint = eyes.mean(axis=0)
            shoulder_width = np.linalg.norm(xy[11]-xy[12])
            head_width = max(np.linalg.norm(xy[7]-xy[8])*1.25, np.linalg.norm(eyes[0]-eyes[1])*2.8, shoulder_width*.36, 28)
            box = (midpoint[0]-head_width*.5, midpoint[1]-head_width*.32, head_width, head_width*1.12)
            wrists = np.array([xy[i] for i in [15,16] if landmarks[i].visibility >= .35], dtype=np.float32)
            reliable=bool(head_width>=60 and np.linalg.norm(eyes[0]-eyes[1])>=18 and min(landmarks[2].visibility,landmarks[5].visibility)>=.6)
            observations.append(Observation(tuple(map(float, box)), eyes=eyes, wrists=wrists,head_reliable=reliable))
        self.face.setInputSize((width,height))
        _, faces = self.face.detect(image)
        clear_faces=[]
        for face in faces if faces is not None else []:
            eyes=face[4:8].reshape(2,2)
            eye_span=float(np.linalg.norm(eyes[0]-eyes[1]))
            if eye_span>.2*face[2] and not any(hand_covers_eye(h,cv2.convexHull(h[PALM]),e,face[2]) for h in self.last_hands for e in eyes):
                clear_faces.append(tuple(map(float,face[:4])))
        flow_heads=self.optical.advance(image,stamp,clear_faces)
        face_rows = []
        used = set()
        for face in faces if faces is not None else []:
            box = tuple(map(float, face[:4]))
            eye_span=float(np.linalg.norm(face[4:6]-face[6:8]))
            if not .18*box[2] < eye_span < .75*box[2]:
                # Occluded false faces can have collapsed eye landmarks.
                continue
            candidates = sorted((distance(box,o.box), i) for i,o in enumerate(observations) if i not in used)
            if candidates and candidates[0][0] < .75:
                index = candidates[0][1]
                obs = observations[index]
                obs.box = box
                obs.eyes = face[4:8].reshape(2,2).copy()
                used.add(index)
            else:
                obs = Observation(box, eyes=face[4:8].reshape(2,2).copy())
                observations.append(obs)
                index = len(observations)-1
                used.add(index)
            obs.face_detected = True
            obs.head_reliable = True
            face_rows.append((obs, face))
        for anchor in flow_heads:
            if any(o.face_detected and distance(o.box,anchor.box)<.65 and o.box[2]>.6*anchor.box[2] for o in observations):
                continue
            nearby=sorted((distance(o.box,anchor.box),i) for i,o in enumerate(observations) if not o.face_detected)
            if nearby and nearby[0][0]<1.1:
                obs=observations[nearby[0][1]]
                obs.box=anchor.box; obs.eyes=anchor.eyes.copy(); obs.head_reliable=True
            else:
                observations.append(Observation(anchor.box,eyes=anchor.eyes.copy(),head_reliable=True))
        unassigned=self.hand_ownership.assign(self.last_hands,observations,stamp)
        for points in unassigned:
            # Never invent a new owner from proximity alone.
            for obs in observations:
                if obs.eyes is not None and min(np.linalg.norm(points[9]-e) for e in obs.eyes)<obs.box[2]:
                    obs.ownership_uncertain=True
        for obs, face in face_rows:
            # Do not enroll covered, small, weak or heavily turned faces.
            clear = not obs.ownership_uncertain and not any(
                hand_covers_eye(h,cv2.convexHull(h[PALM]),e,obs.box[2]) for h in self.last_hands for e in obs.eyes)
            eye_distance = np.linalg.norm(obs.eyes[0]-obs.eyes[1])
            eye_midpoint=obs.eyes.mean(axis=0)
            frontal=abs(float(face[8]-eye_midpoint[0]))<eye_distance*.45
            obs.quality = bool(clear and frontal and face[14]>=.90 and min(obs.box[2:])>=55 and eye_distance>obs.box[2]*.20)
            if obs.quality:
                aligned = self.recognizer.alignCrop(image, face)
                obs.feature = normalize(self.recognizer.feature(aligned))
        self.optical.refresh(observations,stamp)
        self.last_flow_heads=len(flow_heads)
        return observations

    def close(self):
        self.hand.close()
        self.pose.close()
