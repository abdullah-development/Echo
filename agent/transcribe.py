"""Local speech-to-text with faster-whisper, run on a worker thread. Version 2 runs two of
these: a fast model whose text appears instantly, and an accurate one that corrects it."""

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
    def __init__(
        self,
        model_size: str,
        language: str | None,
        on_text: Callable[[int, str, float, float], None],
        cpu_threads: int = 4,
        max_queue: int = 0,
    ):
        print(f"Loading Whisper model '{model_size}' (first run downloads it)...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
        # Multilingual models (e.g. large-v3-turbo) need a language, or they guess per clip.
        self.language = language or (None if model_size.endswith(".en") else "en")
        self.on_text = on_text
        self.max_queue = max_queue  # 0 = unbounded; otherwise skip the oldest clips when behind
        self.prompt: str | None = None  # vocabulary hint, see vocab.py
        self.queue: queue.Queue[tuple[int, np.ndarray, float]] = queue.Queue()
        self._working = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, key: int, audio: np.ndarray, ts: float | None = None) -> None:
        if self.max_queue:
            while self.queue.qsize() >= self.max_queue:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
        self.queue.put((key, audio, ts or time.time()))

    @property
    def busy(self) -> bool:
        return self._working or not self.queue.empty()

    def _run(self) -> None:
        while True:
            key, audio, ts = self.queue.get()
            self._working = True
            try:
                self._transcribe(key, audio, ts)
            finally:
                self._working = False

    def _transcribe(self, key: int, audio: np.ndarray, ts: float) -> None:
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
        if text.lower() in HALLUCINATIONS or text.lower().startswith("glossary"):
            text = ""
        self.on_text(key, text, start, ts)  # empty text: nothing usable was said
