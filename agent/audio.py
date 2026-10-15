"""Audio sources (system audio via the Swift helper, or a file) and a voice-activity
segmenter that cuts the stream into utterances for Whisper as soon as the speaker pauses."""

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16_000
FRAME = SAMPLE_RATE // 10  # 100 ms chunks read from the sources

VAD_WINDOW = 512  # Silero VAD works on 32 ms windows at 16 kHz
VAD_CONTEXT = 64
WINDOWS_PER_S = SAMPLE_RATE / VAD_WINDOW

AUDIOTAP = Path(__file__).resolve().parent.parent / "bin" / "audiotap"


class StreamingVAD:
    """Silero VAD (the model bundled with faster-whisper), run one window at a time.
    It recognises voices rather than loudness, so an open mic's background noise
    doesn't count as speech."""

    def __init__(self):
        from faster_whisper.vad import get_vad_model

        self.session = get_vad_model().session
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(VAD_CONTEXT, dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self.context, window])[None, :].astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = window[-VAD_CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


class Segmenter:
    """Groups VAD windows into utterances: starts when a voice is heard and ends after
    a short pause in speech (or a hard length cap, for non-stop audio)."""

    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        silence_s: float = 0.6,
        max_s: float = 15.0,
        min_speech_s: float = 0.25,
        preroll_s: float = 0.3,
        start_prob: float = 0.5,
        end_prob: float = 0.35,
    ):
        self.on_utterance = on_utterance
        self.vad = StreamingVAD()
        self.silence_windows = int(silence_s * WINDOWS_PER_S)
        self.max_windows = int(max_s * WINDOWS_PER_S)
        self.min_speech_windows = int(min_speech_s * WINDOWS_PER_S)
        self.preroll_windows = int(preroll_s * WINDOWS_PER_S)
        self.start_prob, self.end_prob = start_prob, end_prob

        self.pending = np.zeros(0, dtype=np.float32)
        self.preroll: list[np.ndarray] = []
        self.windows: list[np.ndarray] = []
        self.speech_windows = 0
        self.quiet_run = 0

    @property
    def in_speech(self) -> bool:
        return bool(self.windows)

    def feed(self, samples: np.ndarray) -> None:
        self.pending = np.concatenate([self.pending, samples])
        while len(self.pending) >= VAD_WINDOW:
            window, self.pending = self.pending[:VAD_WINDOW], self.pending[VAD_WINDOW:]
            self._process(window)

    def flush(self) -> None:
        if self.windows:
            self._emit()

    def _process(self, window: np.ndarray) -> None:
        prob = self.vad(window)

        if not self.windows:
            if prob >= self.start_prob:
                self.windows = [*self.preroll, window]
                self.speech_windows, self.quiet_run = 1, 0
            else:
                self.preroll = (self.preroll + [window])[-self.preroll_windows :]
            return

        self.windows.append(window)
        if prob >= self.end_prob:  # hysteresis: once talking, a weaker signal still counts
            self.speech_windows += 1
            self.quiet_run = 0
        else:
            self.quiet_run += 1

        if self.quiet_run >= self.silence_windows or len(self.windows) >= self.max_windows:
            self._emit()

    def _emit(self) -> None:
        windows, speech = self.windows, self.speech_windows
        self.windows, self.preroll, self.speech_windows, self.quiet_run = [], [], 0, 0
        if speech >= self.min_speech_windows:
            self.on_utterance(np.concatenate(windows))


class AudioSource:
    """Runs in a background thread and feeds samples to a Segmenter."""

    def __init__(self, segmenter: Segmenter):
        self.segmenter = segmenter
        self.paused = False
        self.level = 0.0  # last chunk RMS, shown as a meter in the UI
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _feed(self, samples: np.ndarray) -> None:
        if len(samples):
            self.level = float(np.sqrt(np.mean(samples * samples)))
        if not self.paused:
            self.segmenter.feed(samples)

    def _run(self) -> None:
        raise NotImplementedError


class SystemAudioSource(AudioSource):
    def __init__(self, segmenter: Segmenter):
        super().__init__(segmenter)
        if not AUDIOTAP.exists():
            sys.exit(f"Missing {AUDIOTAP}. Build it first: ./build.sh")
        self.proc: subprocess.Popen | None = None

    def _run(self) -> None:
        self.proc = subprocess.Popen([str(AUDIOTAP)], stdout=subprocess.PIPE)
        chunk_bytes = FRAME * 4
        leftover = b""
        while True:
            data = self.proc.stdout.read1(chunk_bytes) if self.proc.stdout else b""
            if not data:
                print("audiotap exited; check Screen Recording permission.", file=sys.stderr)
                return
            data = leftover + data
            usable = len(data) - len(data) % 4
            leftover = data[usable:]
            self._feed(np.frombuffer(data[:usable], dtype="<f4").copy())

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()


class FileSource(AudioSource):
    """Plays an audio/video file through the pipeline in real time - handy for testing."""

    def __init__(self, segmenter: Segmenter, path: str):
        super().__init__(segmenter)
        self.path = path

    def _run(self) -> None:
        import av

        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        with av.open(self.path) as container:
            for frame in container.decode(audio=0):
                chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
        audio = np.concatenate(chunks).astype(np.float32)
        for i in range(0, len(audio), FRAME):
            self._feed(audio[i : i + FRAME])
            time.sleep(0.1)
        # Trailing silence so the last utterance closes.
        self._feed(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
        self.segmenter.flush()

    def stop(self) -> None:
        pass
