import Foundation
import UIKit
import CoreMedia
import MWDATCamera

/// One frame being encoded/sent and one replaceable latest frame. Callback work
/// is constant-time. Network/encoding never runs in the SDK callback or UI actor.
final class FrameTransport: @unchecked Sendable {
    private let lock = NSLock()
    private var latest: (VideoFrame, Int)?
    private var sequence = 0
    private var stopped = false
    private var worker: Task<Void, Never>?
    private let socket: URLSessionWebSocketTask
    private let token: String
    private let report: @Sendable (String) -> Void
    init(url: URL, token: String, report: @escaping @Sendable (String) -> Void) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 5
        configuration.timeoutIntervalForResource = 8
        socket = URLSession(configuration: configuration).webSocketTask(with: url)
        self.token = token; self.report = report
    }
    func connect() async throws {
        socket.resume()
        let auth = try JSONSerialization.data(withJSONObject: ["upload_token": token])
        try await socket.send(.string(String(decoding: auth, as: UTF8.self)))
        guard case let .string(text) = try await receiveBounded(),
              let data = text.data(using: .utf8),
              let ready = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              ready["ready"] as? Bool == true else { throw BridgeFailure.message("Backend rejected camera connection") }
    }
    func offer(_ frame: VideoFrame) {
        lock.withLock {
            guard !stopped else { return }
            sequence += 1; latest = (frame, sequence)
        }
    }
    func start() {
        worker = Task.detached { [self] in
            var sent = 0
            let firstFrameDeadline = Date().addingTimeInterval(8)
            do {
                while !Task.isCancelled {
                    let next = lock.withLock { let value = latest; latest = nil; return value }
                    guard let (frame, number) = next else {
                        if sent == 0 && Date() > firstFrameDeadline {
                            throw BridgeFailure.message("No SDK camera frames within eight seconds. Check camera permission and Mock Device camera availability.")
                        }
                        try await Task.sleep(for: .milliseconds(20)); continue
                    }
                    let sdkMS = CMSampleBufferGetPresentationTimeStamp(frame.sampleBuffer).seconds * 1000
                    guard sdkMS.isFinite, let jpeg = Self.encode(frame), jpeg.count <= 700000 else { continue }
                    let metadata: [String: Any] = ["sequence": number, "sdk_timestamp_ms": sdkMS,
                        "orientation": "up", "provenance": "meta_dat_mock_live_iphone"]
                    let header = try JSONSerialization.data(withJSONObject: metadata)
                    try await socket.send(.string(String(decoding: header, as: UTF8.self)))
                    try await socket.send(.data(jpeg))
                    _ = try await receiveBounded()
                    sent += 1
                    if sent == 1 || sent % 30 == 0 { report("\(sent) real Meta SDK callback frames admitted by backend") }
                }
            } catch {
                if !Task.isCancelled { report("Camera upload disconnected: \(error.localizedDescription). Backend protection remains active; reconnect the bridge.") }
                stop()
            }
        }
    }
    private static func encode(_ frame: VideoFrame) -> Data? {
        autoreleasepool {
            guard let image = frame.makeUIImage() else { return nil }
            let scale = min(1, 960 / max(image.size.width, image.size.height))
            let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
            let format = UIGraphicsImageRendererFormat(); format.scale = 1; format.opaque = true
            // UIImage.draw applies its orientation; the JPEG is baked upright.
            return UIGraphicsImageRenderer(size: size, format: format).image { _ in
                image.draw(in: CGRect(origin: .zero, size: size))
            }.jpegData(compressionQuality: 0.75)
        }
    }
    private func receiveBounded() async throws -> URLSessionWebSocketTask.Message {
        let timeout = Task { [socket] in
            try await Task.sleep(for: .seconds(5))
            socket.cancel(with: .goingAway, reason: nil)
        }
        defer { timeout.cancel() }
        return try await socket.receive()
    }
    func stop() {
        lock.withLock { stopped = true; latest = nil }
        worker?.cancel(); socket.cancel(with: .goingAway, reason: nil)
    }
}
