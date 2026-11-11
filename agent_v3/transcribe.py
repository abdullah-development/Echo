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
    def __init__(self, model_size: str, language: str | None, on_text: Callable[[str, float, float, str], None]):
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
        # The call's audio and your mic are queued separately and the call always goes first,
        # so a noisy room can never hold up the other person's question.
        self.queues: dict[str, queue.Queue[tuple[np.ndarray, float]]] = {"them": queue.Queue(), "you": queue.Queue()}
        self._ready = threading.Semaphore(0)
        self._working: str | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def set_terms(self, terms: list[str]) -> None:
        self.fixer = TermFixer(terms, self.common)

    def submit(self, audio: np.ndarray, speaker: str = "them") -> None:
        self.queues[speaker].put((audio, time.time()))
        self._ready.release()

    def busy_with(self, speaker: str) -> bool:
        return self._working == speaker or not self.queues[speaker].empty()

    @property
    def busy(self) -> bool:
        return self.busy_with("them")

    def _run(self) -> None:
        while True:
            self._ready.acquire()
            speaker = "them" if not self.queues["them"].empty() else "you"
            audio, ts = self.queues[speaker].get()
            self._working = speaker
            try:
                self._transcribe(audio, ts, speaker)
            except Exception as e:  # one bad clip shouldn't stop transcription
                print(f"Transcription failed: {e}")
            finally:
                self._working = None

    def _transcribe(self, audio: np.ndarray, ts: float, speaker: str) -> None:
        start = ts - len(audio) / 16_000
        text = self.model.recognize(audio.astype(np.float32), sample_rate=16_000).strip()
        if text.lower().strip(".!? ") in HALLUCINATIONS:
            if speaker == "you":
                print(f"[mic] nothing recognisable in that clip ({text!r})", flush=True)
            return
        self.on_text(self.fixer.fix(text), start, ts, speaker)
