"""Single frame/session contract. Images are BGR uint8; boxes are pixel xywh internally."""
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp_ms: int
    bgr: np.ndarray
    source_metadata: dict | None = None

@dataclass(frozen=True)
class PrivacyRequest:
    request_id: str
    session_id: str
    source: Literal["gesture", "voice", "test"]
    person_id: str | None
    requested_at_ms: int
    action: Literal["PROTECT_PERSON", "UNPROTECT_PERSON", "SHIELD_UNRESOLVED"]
    reason: str | None = None

@dataclass
class TrackView:
    person_id: str
    privacy: str
    visibility: str
    identity_status: str
    face_box: list[float] | None
    gesture_progress: float
    protection_source: str | None
    last_seen_ms: int
    track_id: str | None = None
    temporary_protection: bool = False

@dataclass
class PrivacyEvent:
    event_id: str
    session_id: str
    type: str
    timestamp_ms: int
    person_id: str | None = None
    source: str = "vision"
    details: dict = field(default_factory=dict)
    schema_version: int = 1

@dataclass
class FrameResult:
    frame_id: int
    timestamp_ms: int
    protected_bgr: np.ndarray
    tracks: list[TrackView]
    events: list[PrivacyEvent]
    output_state: Literal["NORMAL", "SHIELDED", "ERROR"]
    inference_ms: float
    reason: str | None = None
    protection_mode: str = "TARGETED"
