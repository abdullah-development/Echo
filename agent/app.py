"""Live Reply Agent - VERSION 2 (hybrid): listens to system audio, transcribes it, and drafts
replies to questions it hears. A fast Whisper model shows text instantly; an accurate one
re-transcribes each clip a few seconds later, corrects the line, and rewrites the reply if
the correction changed what was asked. Open http://localhost:8766 for the live panel."""

import argparse
import asyncio
import difflib
import itertools
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .audio import FileSource, Segmenter, SystemAudioSource  # noqa: E402
from .responder import Responder, looks_like_question  # noqa: E402
from .transcribe import Transcriber  # noqa: E402
from .vocab import build_vocabulary, whisper_prompt  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
TRANSCRIPT_WINDOW_S = 180  # how much recent conversation Claude sees
DUPLICATE_WINDOW_S = 30
TURN_GAP_S = 3.0  # lines separated by less silence than this are one speaker turn
INCOMPLETE_HOLD_S = 2.5  # wait this long for the rest of a half-finished sentence
SETTLE_S = 0.4
MAX_WAIT_S = 12
REVISE_WINDOW_S = 120  # only rewrite answers this recent when a correction comes in

TRAILING_FILLERS = {
    "the", "a", "an", "about", "of", "to", "for", "with", "and", "or", "but", "in", "on", "at",
    "your", "my", "our", "is", "are", "was", "what", "how", "so", "like", "that", "which", "um", "uh",
}


def meaningful_change(old: str, new: str) -> bool:
    """True if the correction changes what was said: "Cuban eats" -> "Kubernetes" counts;
    punctuation, case, "a"/"the" or near-spellings ("Meilisarch" -> "Meilisearch") don't."""
    a, b = re.findall(r"[a-z0-9']+", old.lower()), re.findall(r"[a-z0-9']+", new.lower())
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal" or not any(len(w) > 3 for w in a[i1:i2] + b[j1:j2]):
            continue
        before, after = "".join(a[i1:i2]), "".join(b[j1:j2])
        if difflib.SequenceMatcher(None, before, after).ratio() < 0.8:
            return True
    return False


def looks_incomplete(text: str) -> bool:
    t = text.strip()
    if t.endswith(("...", "…", "-", ",")) or not t.endswith((".", "?", "!")):
        return True
    words = t.rstrip(".?!").split()
    return bool(words) and words[-1].lower() in TRAILING_FILLERS


class Agent:
    def __init__(self, simulate: str | None):
        self.my_name = os.getenv("MY_NAME", "")
        self.mode = os.getenv("MODE", "meeting")  # meeting | video
        self.auto = os.getenv("AUTO", "questions")  # questions | every | off
        self.clients: set[WebSocket] = set()
        self.transcript: deque[list] = deque()  # [end, text, key]; text is corrected in place
        self.lines: dict[int, list] = {}  # key -> transcript entry
        self.keys = itertools.count(1)
        self.recent_answers: dict[str, float] = {}
        self.ids = itertools.count(1)
        self.turn: list[int] = []  # keys of the current speaker turn's lines, not yet answered
        self.turn_end = 0.0
        self.pending: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.answered: dict[int, dict] = {}  # reply id -> {keys, forced, ts} for revising answers
        self.revisions: dict[int, asyncio.Task] = {}
        self.streaming = False  # the pending auto reply has started showing text
        self.history: dict[int, tuple[str, str]] = {}  # reply id -> (question, answer), shown to Claude

        self.responder = Responder(self.my_name)
        language = os.getenv("LANGUAGE") or None
        self.transcriber = Transcriber(
            os.getenv("WHISPER_MODEL", "base.en"), language, self._on_text,
            cpu_threads=int(os.getenv("FAST_THREADS", "4")),
        )
        # The accurate pass skips the oldest clips if it falls far behind non-stop speech.
        refine_model = os.getenv("REFINE_MODEL", "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
        self.refiner = Transcriber(
            refine_model, language, self._on_refined,
            cpu_threads=int(os.getenv("REFINE_THREADS", "4")), max_queue=4,
        ) if refine_model.lower() not in ("", "off", "none") else None
        self._update_vocabulary()
        self.segmenter = segmenter = Segmenter(self._on_segment, silence_s=float(os.getenv("PAUSE_SECONDS", "0.6")))
        self.source = FileSource(segmenter, simulate) if simulate else SystemAudioSource(segmenter)

    def _update_vocabulary(self) -> None:
        terms = build_vocabulary(self.responder.docs + "\n" + self.responder.profile)
        self.transcriber.prompt = whisper_prompt(terms)
        if self.refiner:
            self.refiner.prompt = self.transcriber.prompt

    # --- plumbing -------------------------------------------------------------
    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.source.start()
        self.loop.create_task(self._meter())
        self.loop.create_task(self._keep_warm())

    async def _keep_warm(self) -> None:
        # Keeps a booted claude process ready for the current mode (and recycles old ones).
        while True:
            try:
                await self.responder.prewarm(self.mode)
            except Exception as e:
                print(f"Couldn't pre-start Claude Code: {e}")
            await asyncio.sleep(30)

    async def send(self, msg: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                self.clients.discard(ws)

    def state(self) -> dict:
        return {
            "type": "state",
            "mode": self.mode,
            "auto": self.auto,
            "paused": self.source.paused,
            "docs": self.responder.doc_names,
            "name": self.my_name,
            "model": self.responder.model or "Claude Code",
            "version": "v2 hybrid" if self.refiner else "v2",
        }

    async def _meter(self) -> None:
        while True:
            await self.send({"type": "level", "value": self.source.level})
            await asyncio.sleep(0.2)

    # --- pipeline -------------------------------------------------------------
    def _on_segment(self, audio) -> None:
        """Called by the segmenter for each finished clip of speech."""
        key, ts = next(self.keys), time.time()
        self.transcriber.submit(key, audio, ts)
        if self.refiner:
            self.refiner.submit(key, audio, ts)

    def _on_text(self, key: int, text: str, start: float, end: float) -> None:
        """Called from the fast transcriber thread."""
        assert self.loop
        if text:
            asyncio.run_coroutine_threadsafe(self._handle_text(key, text, start, end), self.loop)

    def _on_refined(self, key: int, text: str, start: float, end: float) -> None:
        """Called from the accurate transcriber thread."""
        assert self.loop
        asyncio.run_coroutine_threadsafe(self._handle_refined(key, text, start, end), self.loop)

    def _text(self, keys: list[int]) -> str:
        return " ".join(self.lines[k][1] for k in keys if k in self.lines)

    async def _handle_text(self, key: int, text: str, start: float, end: float) -> None:
        entry = [end, text, key]
        self.transcript.append(entry)
        self.lines[key] = entry
        while self.transcript and self.transcript[0][0] < time.time() - TRANSCRIPT_WINDOW_S:
            self.lines.pop(self.transcript.popleft()[2], None)
        await self.send({"type": "transcript", "key": key, "text": text, "ts": start,
                         "refining": self.refiner is not None})

        # People often pause mid-question ("can you tell me about the..." ... "testing in
        # OnlineCook"), so consecutive lines form one turn that is answered as a whole.
        if start - self.turn_end > TURN_GAP_S:
            self.turn = []
        self.turn.append(key)
        self.turn_end = end

        # Newer speech supersedes a queued or in-flight auto reply for the same turn.
        if self.pending and not self.pending.done():
            self.pending.cancel()
        if self.auto != "off" and not self.source.paused:
            self.pending = asyncio.create_task(self._answer_turn())

    async def _handle_refined(self, key: int, text: str, start: float, end: float) -> None:
        entry = self.lines.get(key)
        if entry is None:
            # The fast model dropped this clip (unclear audio) but the accurate one heard speech.
            if text:
                await self._handle_text(key, text, start, end)
            return
        old = entry[1]
        if not text or text == old:  # accurate model heard nothing: keep the fast text
            await self.send({"type": "transcript_update", "key": key, "text": old})
            return
        entry[1] = text
        await self.send({"type": "transcript_update", "key": key, "text": text})
        if not meaningful_change(old, text):
            return
        if key in self.turn and self.pending and not self.pending.done():
            if self.streaming:
                # Don't throw away an answer that's already appearing; rewrite it once it's done.
                asyncio.create_task(self._revise_after(self.pending, key))
            else:
                # Still waiting to answer: just answer the corrected question instead.
                self.pending.cancel()
                self.pending = asyncio.create_task(self._answer_turn())
            return
        self._revise(key)

    async def _revise_after(self, task: asyncio.Task, key: int) -> None:
        try:
            await asyncio.shield(task)
        except BaseException:
            return
        self._revise(key)

    def _revise(self, key: int) -> None:
        """Rewrites, in place, any recent answer whose question contained this corrected line."""
        for rid, info in list(self.answered.items()):
            if key in info["keys"] and time.time() - info["ts"] < REVISE_WINDOW_S:
                if rid in self.revisions and not self.revisions[rid].done():
                    self.revisions[rid].cancel()
                self.revisions[rid] = asyncio.create_task(
                    self._reply(self._text(info["keys"]), info["forced"], info["keys"], rid=rid)
                )

    async def _answer_turn(self) -> None:
        self.streaming = False
        await asyncio.sleep(INCOMPLETE_HOLD_S if looks_incomplete(self._text(self.turn[-1:])) else SETTLE_S)
        # Don't answer while they're still talking or the next line is still being transcribed.
        deadline = time.time() + MAX_WAIT_S
        while (self.segmenter.in_speech or self.transcriber.busy) and time.time() < deadline:
            await asyncio.sleep(0.1)

        keys = self.turn[-4:]
        focus = self._text(keys)
        if self.auto == "questions" and not looks_like_question(focus, self.my_name):
            return
        key = focus.lower().strip()
        if time.time() - self.recent_answers.get(key, 0) < DUPLICATE_WINDOW_S:
            return
        self.recent_answers[key] = time.time()
        if await self._reply(focus, forced=False, keys=keys):
            self.turn = []  # answered; a follow-up starts a fresh turn

    async def _reply(self, focus: str, forced: bool, keys: list[int] | None = None, rid: int | None = None) -> bool:
        """Returns True if a reply was written. With rid, rewrites that existing answer."""
        revising = rid is not None
        rid = rid if revising else next(self.ids)
        lines = [t for _, t, _ in self.transcript]
        previous = [qa for r, qa in sorted(self.history.items()) if r != rid][-3:]
        started = False

        async def on_delta(delta: str) -> None:
            nonlocal started
            if revising and not started:  # keep the old answer on screen until the new one starts
                await self.send({"type": "reply_revise", "id": rid, "question": focus})
            started = True
            if not forced and not revising:
                self.streaming = True
            await self.send({"type": "reply_delta", "id": rid, "delta": delta})

        if not revising:
            await self.send({"type": "reply_start", "id": rid, "question": focus, "forced": forced})
        try:
            result = await self.responder.reply(lines, focus, self.mode, on_delta, forced=forced, previous=previous)
        except asyncio.CancelledError:
            if not revising:
                await self.send({"type": "reply_cancel", "id": rid})
            raise
        except Exception as e:  # e.g. claude not signed in, usage limit reached
            await self.send({"type": "reply_error", "id": rid, "error": f"Claude Code: {e}"})
            return False
        if revising and result is None:
            return False  # the corrected question needs no reply: leave the earlier answer as is
        await self.send({"type": "reply_end", "id": rid, "skipped": result is None})
        if result is not None:
            self.history[rid] = (focus, result)
            for old in sorted(self.history)[:-5]:
                self.history.pop(old)
        if result is not None and keys:
            self.answered[rid] = {"keys": keys, "forced": forced, "ts": time.time()}
            for old in [r for r, i in self.answered.items() if time.time() - i["ts"] > REVISE_WINDOW_S]:
                self.answered.pop(old)
        return result is not None

    async def command(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "answer_now" and self.transcript:
            # Answer the last ~2 utterances: questions often span a segment boundary.
            keys = [k for _, _, k in list(self.transcript)[-2:]]
            await self._reply(self._text(keys), forced=True, keys=keys)
        elif kind == "ask" and msg.get("text", "").strip():
            await self._reply(msg["text"].strip(), forced=True)
        elif kind == "set_mode" and msg.get("mode") in ("meeting", "video"):
            self.mode = msg["mode"]
            asyncio.create_task(self.responder.prewarm(self.mode))
        elif kind == "set_auto" and msg.get("auto") in ("questions", "every", "off"):
            self.auto = msg["auto"]
        elif kind == "pause":
            self.source.paused = True
        elif kind == "resume":
            self.source.paused = False
        elif kind == "reload_docs":
            self.responder.reload()
            self._update_vocabulary()
            asyncio.create_task(self.responder.prewarm(self.mode))
        elif kind == "clear":
            self.transcript.clear()
            self.lines.clear()
            self.history.clear()
        await self.send(self.state())


def build_app(agent: Agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        agent.start()
        yield
        agent.source.stop()
        agent.responder.drop_warm()

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        agent.clients.add(socket)
        await socket.send_json(agent.state())
        for ts, text, key in agent.transcript:
            await socket.send_json({"type": "transcript", "key": key, "text": text, "ts": ts})
        try:
            while True:
                asyncio.create_task(agent.command(await socket.receive_json()))
        except WebSocketDisconnect:
            agent.clients.discard(socket)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", metavar="FILE", help="feed an audio/video file instead of system audio")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8766")))
    args = parser.parse_args()

    agent = Agent(args.simulate)
    print(f"\n  Live Reply Agent → http://localhost:{args.port}\n")
    uvicorn.run(build_app(agent), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
