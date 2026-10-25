import SwiftUI
import AVFoundation
import MWDATCore
import MWDATCamera
import MWDATMockDevice

@main
struct ConsentLensBridgeApp: App {
    @StateObject private var bridge = Bridge()
    @Environment(\.scenePhase) private var phase
    var body: some Scene {
        WindowGroup {
            Form {
                Section("ConsentLens camera bridge") {
                    Text("Actual Meta SDK · simulated glasses · live iPhone camera")
                    TextField("Mac URL, e.g. http://192.168.1.20:8000", text: $bridge.address)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Local upload token (not a Meta API key)", text: $bridge.token)
                    Toggle("Use front camera", isOn: $bridge.front)
                    Button(bridge.running ? "Stop camera bridge" : "Connect live iPhone camera") {
                        if bridge.running { bridge.stop() } else { Task { await bridge.start() } }
                    }
                    Text(bridge.status).accessibilityLabel("Bridge status: \(bridge.status)")
                }
                Text("Watch protected output and record both videos in the existing Mac dashboard. Phone microphone audio is not transported. Keep this app in the foreground.")
            }
            .task {
                if ProcessInfo.processInfo.environment["CONSENTLENS_AUTO_CONNECT"] == "1" { await bridge.start() }
            }
            // Permission dialogs temporarily make the app inactive. Only actual
            // backgrounding should tear down an in-progress camera connection.
            .onChange(of: phase) { _, value in if value == .background { bridge.stop() } }
        }
    }
}

@MainActor
final class Bridge: ObservableObject {
    @Published var address = ProcessInfo.processInfo.environment["CONSENTLENS_BACKEND_URL"] ?? "http://"
    @Published var token = ProcessInfo.processInfo.environment["CONSENTLENS_UPLOAD_TOKEN"] ?? ""
    @Published var front = false
    @Published var running = false
    @Published var status = "Not connected"
    private var configured = false
    private var device: (any MockGlasses)?
    private var session: DeviceSession?
    private var camera: MWDATCamera.Camera?
    private var listeners: [any AnyListenerToken] = []
    private var transport: FrameTransport?

    func start() async {
        guard !running else { return }
        guard var parts = URLComponents(string: address),
              let host = parts.host, !["localhost", "127.0.0.1", "::1"].contains(host),
              ["http", "https"].contains(parts.scheme ?? ""), !token.isEmpty else {
            status = "Enter the Mac’s reachable LAN URL and local upload token"; return
        }
        running = true; status = "Requesting iPhone camera access…"
        guard await AVCaptureDevice.requestAccess(for: .video), running else {
            status = "Camera permission is required in iPhone Settings"; running = false; return
        }
        do {
            if !configured { try Wearables.configure(); configured = true }
            MockDeviceKit.shared.enable()
            let mock = try MockDeviceKit.shared.pairGlasses(model: .rayBanMeta)
            device = mock
            mock.services.camera.setCameraFeed(cameraFacing: front ? .front : .back)
            mock.powerOn(); mock.unfold(); mock.don()
            let deadline = Date().addingTimeInterval(8)
            while !Wearables.shared.devices.contains(mock.deviceIdentifier) {
                guard running, Date() < deadline else { throw BridgeFailure.message("Mock device discovery timed out") }
                try await Task.sleep(for: .milliseconds(50))
            }
            let connection = try Wearables.shared.createSession(deviceSelector: SpecificDeviceSelector(device: mock.deviceIdentifier))
            session = connection; try connection.start()
            while connection.state != .started {
                guard running, Date() < deadline else { throw BridgeFailure.message("Meta device session did not start") }
                try await Task.sleep(for: .milliseconds(50))
            }
            guard let capability = try connection.addCamera(config: StreamConfiguration(videoCodec: .raw, audioCodec: nil, resolution: .low, frameRate: 15)) else {
                throw BridgeFailure.message("Meta camera capability unavailable")
            }
            camera = capability
            parts.scheme = parts.scheme == "https" ? "wss" : "ws"
            parts.path = "/api/source/iphone/frames"; parts.query = nil; parts.fragment = nil
            guard let url = parts.url else { throw BridgeFailure.message("Invalid Mac URL") }
            let sender = FrameTransport(url: url, token: token) { [weak self] text in
                Task { @MainActor in self?.status = text }
            }
            transport = sender
            try await sender.connect()
            // This is the SDK's normal camera-stream callback. No direct
            // AVCaptureSession, screen mirroring, movie, or simulated frame source.
            listeners.append(capability.stream.videoFramePublisher.listen { frame in sender.offer(frame) })
            listeners.append(capability.stream.errorPublisher.listen { [weak self] error in
                Task { @MainActor in self?.status = "Meta stream error: \(error)"; self?.stop(keepStatus: true) }
            })
            capability.stream.start()
            sender.start()
            status = "Meta stream started; waiting for first SDK camera callback…"
        } catch {
            status = "Bridge failed: \(error.localizedDescription)"
            stop(keepStatus: true)
        }
    }

    func stop(keepStatus: Bool = false) {
        running = false
        transport?.stop(); transport = nil
        camera?.stop(); camera = nil
        session?.stop(); session = nil
        let retained = listeners; listeners.removeAll()
        Task { for listener in retained { await listener.cancel() } }
        MockDeviceKit.shared.disable(); device = nil
        if !keepStatus { status = "Stopped; backend privacy memory remains active" }
    }
}

enum BridgeFailure: LocalizedError {
    case message(String)
    var errorDescription: String? { if case let .message(text) = self { return text }; return nil }
}
