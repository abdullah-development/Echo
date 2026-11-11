"""Local speech-to-text with NVIDIA Parakeet TDT 0.6B v3 (int8, ONNX, CPU), run on a
worker thread. More accurate than Whisper large-v3-turbo and faster than base.en on this
Mac; it has no vocabulary prompt, so misheard jargon is fixed afterwards (termfix.py)."""

import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnx_asr

from .termfix import TermFixer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "parakeet-tdt-0.6b-v3-int8"
COMMON_WORDS = Path("/usr/share/dict/words")

# Filler a speech model may emit for noise or music.
HALLUCINATIONS = {"", "you", "uh", "um", "mm", "hmm", "[music]", "(music)", "[silence]"}


class Transcriber:
    def __init__(self, model_size: str, language: str | None, on_text: Callable[[str, float, float], None]):
        print("Loading Parakeet v3 speech model...")
        # Plain CPU: Apple's CoreML provider only covers part of the graph and produces garbage.
        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3", str(MODEL_DIR), quantization="int8", providers=["CPUExecutionProvider"]
        )
        self.on_text = on_text
        self.prompt: str | None = None  # unused (Whisper only); kept so the app code is shared
        self.common = (
            {w.strip().lower() for w in COMMON_WORDS.read_text().split()} if COMMON_WORDS.exists() else set()
        )
        self.fixer = TermFixer([], self.common)
        self.queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()
        self._working = False
        threading.Thread(target=self._run, daemon=True).start()

    def set_terms(self, terms: list[str]) -> None:
        self.fixer = TermFixer(terms, self.common)

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
            except Exception as e:  # one bad clip shouldn't stop transcription
                print(f"Transcription failed: {e}")
            finally:
                self._working = False

    def _transcribe(self, audio: np.ndarray, ts: float) -> None:
        start = ts - len(audio) / 16_000
        text = self.model.recognize(audio.astype(np.float32), sample_rate=16_000).strip()
        if text.lower().strip(".!? ") in HALLUCINATIONS:
            return
        self.on_text(self.fixer.fix(text), start, ts)
