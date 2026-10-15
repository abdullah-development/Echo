"""Local speech-to-text with faster-whisper, run on a worker thread."""

import queue
import threading
import time
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel

# Whisper tends to "hear" these in silence or music.
HALLUCINATIONS = {
    "", "you", "thank you.", "thanks for watching!", "thank you for watching.",
    "thanks for watching.", "bye.", "[music]", "(music)", "[silence]",
}


class Transcriber:
    def __init__(self, model_size: str, language: str | None, on_text: Callable[[str, float, float], None]):
        print(f"Loading Whisper model '{model_size}' (first run downloads it)...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.language = language
        self.on_text = on_text
        self.prompt: str | None = None  # vocabulary hint, see vocab.py
        self.queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()
        self._working = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, audio: np.ndarray) -> None:
        self.queue.put((audio, time.time()))

    @property
    def busy(self) -> bool:
        return self._working or not self.queue.empty()

    def _run(self) -> None:
        while True:
            audio, ts = self.queue.get()
            self._working = True
            try:
                self._transcribe(audio, ts)
            finally:
                self._working = False

    def _transcribe(self, audio: np.ndarray, ts: float) -> None:
        start = ts - len(audio) / 16_000
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.prompt,
        )
        # Drop low-confidence segments: on unclear audio Whisper tends to "fill in"
        # words from the vocabulary prompt (e.g. "Bye, Kibana").
        text = " ".join(
            s.text.strip() for s in segments if s.no_speech_prob < 0.5 and s.avg_logprob > -1.0
        ).strip()
        # (the glossary check drops rare echoes of the vocabulary prompt on near-silence)
        if text.lower() not in HALLUCINATIONS and not text.lower().startswith("glossary"):
            self.on_text(text, start, ts)
