// audiotap: captures macOS system audio with ScreenCaptureKit and writes it to
// stdout as raw 16 kHz mono float32 little-endian PCM.
//
// Requires Screen Recording permission for the app that launches it
// (System Settings → Privacy & Security → Screen & System Audio Recording).

import CoreMedia
import Foundation
import ScreenCaptureKit

let sampleRate = 16_000

final class AudioSink: NSObject, SCStreamOutput, SCStreamDelegate {
    private let out = FileHandle.standardOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        try? sampleBuffer.withAudioBufferList { buffers, _ in
            // channelCount = 1, so the first (non-interleaved) buffer holds the mono samples.
            guard let buf = buffers.first, let data = buf.mData, buf.mDataByteSize > 0 else { return }
            out.write(Data(bytes: data, count: Int(buf.mDataByteSize)))
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("audiotap: stream stopped: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(1)
    }
}

let sink = AudioSink()
var activeStream: SCStream?

signal(SIGPIPE, SIG_DFL)  // exit quietly when the reader goes away

Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            FileHandle.standardError.write("audiotap: no display found\n".data(using: .utf8)!)
            exit(1)
        }
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = sampleRate
        config.channelCount = 1
        // Video is mandatory for SCStream; keep it as small and infrequent as possible.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(filter: filter, configuration: config, delegate: sink)
        try stream.addStreamOutput(sink, type: .audio, sampleHandlerQueue: DispatchQueue(label: "audiotap.audio"))
        try await stream.startCapture()
        activeStream = stream
        FileHandle.standardError.write("audiotap: capturing system audio at \(sampleRate) Hz\n".data(using: .utf8)!)
    } catch {
        FileHandle.standardError.write("audiotap: \(error.localizedDescription)\n".data(using: .utf8)!)
        FileHandle.standardError.write("audiotap: grant Screen Recording permission to your terminal and retry\n".data(using: .utf8)!)
        exit(1)
    }
}

dispatchMain()
