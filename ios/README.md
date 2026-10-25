# iPhone bridge setup

Actual Meta Device Access Toolkit **1.0.1**, simulated glasses, **live iPhone camera**. SDK callback frames enter the existing Python engine; no direct-camera bypass or prerecorded feed is used. Phone audio is not transported.

1. Open `ConsentLensBridge.xcodeproj` in Xcode. Resolve the pinned package, install support for your iOS version, select your own signing team and a unique bundle identifier if needed. The public project deliberately has no personal signing team.
2. Connect an unlocked iPhone with a data cable, trust the Mac, enable Developer Mode, and select it as the run destination. Complete Apple sign-in, developer trust and app verification locally. Never share credentials.
3. Start the backend from the repository root:
   ```sh
   CONSENTLENS_CAMERA_SOURCE=iphone CONSENTLENS_BIND=0.0.0.0 .venv/bin/python run.py
   ```
4. Open `http://127.0.0.1:8000` on the Mac, enter the dashboard, select iPhone and start a session. Under the settings gear, open iPhone bridge setup and copy the **local upload token**. It is not a Meta API key and changes after a backend restart.
5. Put the phone and Mac on a mutually reachable trusted network. Run `scutil --get LocalHostName` on the Mac. Enter `http://<MAC-HOSTNAME>.local:8000` in the bridge, plus the upload token. Do not use localhost on the phone.
6. This public configuration permits local hostname networking. If your network requires a numeric IP instead, obtain it with `ipconfig getifaddr en0`, add an exact local development `NSExceptionDomains` entry with `NSExceptionAllowsInsecureHTTPLoads=true` under `NSAppTransportSecurity` in your local `Info.plist`, and rebuild. The original developer's IP exception was removed from the public copy. Do not broadly disable transport security or publish your personal settings.
7. Install/run, grant Camera and Local Network access, and tap Connect. Keep the app foregrounded and unlocked. It defaults to the rear camera. Confirm the admitted SDK-frame count increases and moving the phone changes the protected web preview.

Optional configured launch, after installation and starting the backend session:

```sh
.venv/bin/python scripts/launch_iphone_bridge.py --device <CONNECTED-DEVICE-ID> --backend-url http://<MAC-HOSTNAME>.local:8000
```

The helper privately forwards the local upload configuration through devicectl child environment variables. It does not reset the session or read the Meta Model API key.

The bridge configures Mock Device Kit's live phone camera, creates a device session and camera capability, and subscribes to `videoFramePublisher`. One latest-frame mailbox bounds queued work. JPEG encoding applies image orientation; the backend letterboxes frames. Five-second handshake/acknowledgment bounds prevent indefinite waits. Source errors/disconnects preserve privacy preferences, but camera reconnection must succeed before a live demonstration can proceed.

Physical iPhone SDK-to-web delivery, one-person gestures and paired recording were exercised during development. This does not establish reliability on every phone/network or physical glasses. SDK provenance metadata is an application label, not cryptographic attestation. Native Meta recording enforcement is not implemented.

Upstream: [Meta iOS toolkit](https://github.com/facebook/meta-wearables-dat-ios/tree/1.0.1). SDK developer configuration uses `MetaAppID=0` and an empty client token with Mock Device Kit; neither is a Model API credential. Configure real-device/product access separately when moving beyond this prototype.
