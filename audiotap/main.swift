// audiotap: captures macOS system audio with ScreenCaptureKit and writes it to
// stdout as raw 16 kHz mono float32 little-endian PCM.
// `audiotap --mic` captures the default microphone instead, in the same format.
//
// Requires Screen Recording permission for the app that launches it
// (System Settings → Privacy & Security → Screen & System Audio Recording),
// or Microphone permission for --mic.

import AVFoundation
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

func fail(_ message: String) -> Never {
    FileHandle.standardError.write("audiotap: \(message)\n".data(using: .utf8)!)
    exit(1)
}

let sink = AudioSink()
var activeStream: SCStream?
let engine = AVAudioEngine()

signal(SIGPIPE, SIG_DFL)  // exit quietly when the reader goes away

func captureMicrophone() {
    let input = engine.inputNode
    let inFormat = input.outputFormat(forBus: 0)
    guard inFormat.sampleRate > 0 else { fail("no microphone found") }
    let outFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
    guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else { fail("can't convert microphone audio") }
    let out = FileHandle.standardOutput

    input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { buffer, _ in
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * outFormat.sampleRate / inFormat.sampleRate) + 32
        guard let converted = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
        var fed = false
        var error: NSError?
        converter.convert(to: converted, error: &error) { _, status in
            if fed {
                status.pointee = .noDataNow
                return nil
            }
            fed = true
            status.pointee = .haveData
            return buffer
        }
        if let samples = converted.floatChannelData, converted.frameLength > 0 {
            out.write(Data(bytes: samples[0], count: Int(converted.frameLength) * 4))
        }
    }
    do {
        try engine.start()
    } catch {
        fail("microphone: \(error.localizedDescription)")
    }
    FileHandle.standardError.write("audiotap: capturing microphone at \(sampleRate) Hz\n".data(using: .utf8)!)
}

if CommandLine.arguments.contains("--mic") {
    AVCaptureDevice.requestAccess(for: .audio) { granted in
        guard granted else { fail("grant Microphone permission to your terminal and retry") }
        DispatchQueue.main.async { captureMicrophone() }
    }
} else { Task {
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
} }

dispatchMain()
