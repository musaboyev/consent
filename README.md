# Consent

**Your visibility. Your choice.** A HackTX prototype of bystander-controlled face protection for wearable-camera video.

Cover both eyes deliberately to blur your face; make a peace sign to reveal it. Spoken requests update the same person's privacy preference. Lowering your hands does not remove protection. Preferences are remembered within the session when you leave and return; an empty scene stays clear.

## Actual integration

**Live iPhone camera → Meta iOS Device Access Toolkit / Mock Device Kit → Swift WebSocket bridge → local Python privacy engine → protected web preview and simultaneous original/protected MP4 recordings.**

The bridge uses the actual Meta SDK (pinned to 1.0.1), simulated glasses, and the live iPhone camera delivered through the SDK camera callback. Physical Meta glasses were unavailable. This is a local demonstration, not enforcement across native Meta recordings, a production privacy guarantee, or a crime-detection system. The webcam is a development fallback.

The backend uses OpenCV YuNet/SFace, MediaPipe hand/pose landmarks, separate tracking and session identities, and pixel-level facial blur. The React dashboard displays the protected pixels. Original recordings are explicitly labeled unprotected developer diagnostics. No recording audio is included.

The **Mac microphone** feeds real **Muse Voice Transcribe**, then **Muse Spark** interprets the actual transcript into validated protect/reveal/no-action results. The phone does not send audio. Voice assumes exactly one visible person is speaking; multi-person speaker-to-face identification is not implemented. “Don't film me” requests face blur, not stopping recording. Silence or absence of an objection is never labeled consent.

## Install

Tested during development on Apple Silicon macOS with Python 3.13.7. Use Node.js compatible with Vite 7 (Node 22.12+ is suitable) and Xcode with iOS device support for the phone bridge.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python scripts/download_models.py
npm --prefix frontend ci
npm --prefix frontend run build
cp .env.example .env
```

Model downloads are verified against `models/manifest.json`. We do not distribute model binaries; upstream model licenses are listed in `models/MANIFEST.md`. The dependency lock records the development environment; availability on other platforms must be checked.

Edit `.env` locally and set `MODEL_API_KEY` if using Meta voice. The example has placeholders only. Do not put this key in the frontend, iOS app, screenshots, or Git. The existing model defaults are `muse-voice-transcribe-1.0` and `muse-spark-1.3`; actual access depends on your Meta account.

```sh
# Optional small real Meta access checks; these consume the shared budget.
.venv/bin/python scripts/probe_meta.py

# Main iPhone demonstration server: one process, one worker.
CONSENTLENS_CAMERA_SOURCE=iphone CONSENTLENS_BIND=0.0.0.0 .venv/bin/python run.py
```

Open **http://127.0.0.1:8000** on the Mac. Select the iPhone source and start a camera session. Follow [iPhone setup](ios/README.md) to connect the bridge. Do not start a second backend if one is already running.

For webcam development, run `.venv/bin/python run.py` without those environment overrides. Allow macOS Camera/Microphone access for the app launching Python. The browser does not own the camera or microphone.

## Demonstrate

1. Keep the phone app foregrounded and unlocked, with face, shoulders, hand and wrist visible in good light. Confirm the preview is moving.
2. Wait briefly with hands down for identity enrollment. Cover **both eyes** steadily for about one second. Lowering the hand leaves the face blurred.
3. Leave completely, then return facing the camera. The background stays clear during absence and the recognized person's preference is restored.
4. Hold a clear V/peace sign beside the face for about one second to reveal that person. Either hand can be used; cropped or ambiguously associated hands may be rejected.
5. With only one person visible, enable Meta voice and speak toward the **Mac microphone**. Try “Blur my face” and “Unblur my face,” pausing for each transcript/action.
6. Start recording, perform the sequence, then stop and finalize. Separate Protected and Original / unprotected Play/Download controls become available.

Recording stop/start and dashboard reconnect do not reset privacy state. **End session & clear memory** deliberately does. Source disconnect retains preferences but shows an error slate until frames resume. Keep the app open; backgrounding stops the bridge.

## Verification and limitations

Live development trials demonstrated iPhone SDK-to-web frames, automatic eye-cover protection, peace-sign reveal, either-hand operation when fully visible, remembered return preferences, and playable paired videos with different protected pixels. Participants also reported successful two-person gesture operation after corrections; this is not a comprehensive crowd/crossing accuracy evaluation.

English spoken controls were exercised. Russian and Spanish spoken behavior was observed/reported in demo trials; broader speech support remains experimental. Mandarin had successful spoken reveal trials (including `请帮我去除模糊`) but also incomplete or incorrect transcripts. Written multilingual intent tests do not establish spoken reliability or a guaranteed four-language sequence. Negated reveal requests must not reveal a face; unclear or invalid output is not treated as permission to reveal.

Identity can fail with poor light, small/profile faces, occlusion, crossings, or camera motion. Unresolved located faces can be temporarily masked; a valid owned peace sign requires sufficient identity evidence. Pending requests are not silently cleared for other people. A completely missed face cannot be protected by face detection. Facial blur does not anonymize clothing, voice, or surrounding context.

Gestures require both spatial evidence and multiple consecutive valid observations; response time varies with incoming frame cadence. No general accuracy or FPS guarantee is claimed. Original input is preserved for its encoder, but MP4 is lossy. Source outages create timestamp gaps; no frames are synthesized during an outage. Stop finalizes live encoders, with no offline blur pass.

## Tests

```sh
.venv/bin/python scripts/smoke.py
.venv/bin/python -m pytest -q
```

Tests cover privacy state, both-eye geometry, gesture timing, pending-request recovery, person separation, phone transport, reconnection, recordings, structured voice intent, and budget accounting. Fixtures and unit tests are not fresh hardware trials. Private participant footage and local test logs are intentionally excluded from this repository.

## Meta budget and data handling

A persistent **$20 estimated application-side budget** in `.local/meta-budget.sqlite3` is shared across Meta requests, including probes. It reserves costs before requests and reconciles returned usage where available. Output, retries, and streaming duration are bounded. Exhaustion disables Meta features while local vision continues. This is not an account-wide billing cap; missing usage is conservatively estimated. Preserve the ledger across restarts. Pricing assumptions are recorded in `backend/meta/pricing.json`.

Camera frames are processed locally after phone transport. Microphone audio and transcripts go to Meta when voice is enabled. API keys remain backend-only. The phone uses a separate local upload token. LAN development transport is unencrypted; use a trusted network. Dashboard and diagnostic recording access are loopback-restricted; this prototype is not a secure evidence vault.

Recordings, credentials, session galleries/recovery snapshots, budget databases, local logs, signing configuration, downloaded models and build outputs are excluded from publication. Session recovery files may contain biometric data and must remain private.

## Development tools and licensing

The team used **OpenAI Codex and Anthropic Claude** as AI-assisted development tools. These are development tools, not substitute runtime providers for the Meta voice integration. Human trials and code checks were used to assess behavior; limitations above remain.

No project software license has been selected. This publication does not add one. Third-party dependencies, SDKs, and models retain their respective licenses and terms.
