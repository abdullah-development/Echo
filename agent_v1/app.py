"""Live Reply Agent - VERSION 1 (frozen): base.en only. Listens to system audio, transcribes it, and drafts replies to
questions it hears. Open http://localhost:8766 for the live panel."""

import argparse
import asyncio
import itertools
import os
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

TRAILING_FILLERS = {
    "the", "a", "an", "about", "of", "to", "for", "with", "and", "or", "but", "in", "on", "at",
    "your", "my", "our", "is", "are", "was", "what", "how", "so", "like", "that", "which", "um", "uh",
}


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
        self.transcript: deque[tuple[float, str]] = deque()
        self.recent_answers: dict[str, float] = {}
        self.ids = itertools.count(1)
        self.turn: list[str] = []  # lines of the current speaker turn, not yet answered
        self.turn_end = 0.0
        self.pending: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

        self.responder = Responder(self.my_name)
        self.transcriber = Transcriber(
            os.getenv("WHISPER_MODEL", "base.en"), os.getenv("LANGUAGE") or None, self._on_text
        )
        self._update_vocabulary()
        self.segmenter = segmenter = Segmenter(self.transcriber.submit, silence_s=float(os.getenv("PAUSE_SECONDS", "0.6")))
        self.source = FileSource(segmenter, simulate) if simulate else SystemAudioSource(segmenter)

    def _update_vocabulary(self) -> None:
        terms = build_vocabulary(self.responder.docs + "\n" + self.responder.profile)
        self.transcriber.prompt = whisper_prompt(terms)

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
        }

    async def _meter(self) -> None:
        while True:
            await self.send({"type": "level", "value": self.source.level})
            await asyncio.sleep(0.2)

    # --- pipeline -------------------------------------------------------------
    def _on_text(self, text: str, start: float, end: float) -> None:
        """Called from the transcriber thread."""
        assert self.loop
        asyncio.run_coroutine_threadsafe(self._handle_text(text, start, end), self.loop)

    async def _handle_text(self, text: str, start: float, end: float) -> None:
        self.transcript.append((end, text))
        while self.transcript and self.transcript[0][0] < time.time() - TRANSCRIPT_WINDOW_S:
            self.transcript.popleft()
        await self.send({"type": "transcript", "text": text, "ts": start})

        # People often pause mid-question ("can you tell me about the..." ... "testing in
        # OnlineCook"), so consecutive lines form one turn that is answered as a whole.
        if start - self.turn_end > TURN_GAP_S:
            self.turn = []
        self.turn.append(text)
        self.turn_end = end

        # Newer speech supersedes a queued or in-flight auto reply for the same turn.
        if self.pending and not self.pending.done():
            self.pending.cancel()
        if self.auto != "off" and not self.source.paused:
            self.pending = asyncio.create_task(self._answer_turn())

    async def _answer_turn(self) -> None:
        await asyncio.sleep(INCOMPLETE_HOLD_S if looks_incomplete(self.turn[-1]) else SETTLE_S)
        # Don't answer while they're still talking or the next line is still being transcribed.
        deadline = time.time() + MAX_WAIT_S
        while (self.segmenter.in_speech or self.transcriber.busy) and time.time() < deadline:
            await asyncio.sleep(0.1)

        focus = " ".join(self.turn[-4:])
        if self.auto == "questions" and not looks_like_question(focus, self.my_name):
            return
        key = focus.lower().strip()
        if time.time() - self.recent_answers.get(key, 0) < DUPLICATE_WINDOW_S:
            return
        self.recent_answers[key] = time.time()
        if await self._reply(focus, forced=False):
            self.turn = []  # answered; a follow-up starts a fresh turn

    async def _reply(self, focus: str, forced: bool) -> bool:
        """Returns True if a reply was written."""
        rid = next(self.ids)
        lines = [t for _, t in self.transcript]

        async def on_delta(delta: str) -> None:
            await self.send({"type": "reply_delta", "id": rid, "delta": delta})

        await self.send({"type": "reply_start", "id": rid, "question": focus, "forced": forced})
        try:
            result = await self.responder.reply(lines, focus, self.mode, on_delta, forced=forced)
        except asyncio.CancelledError:
            await self.send({"type": "reply_cancel", "id": rid})
            raise
        except Exception as e:  # e.g. claude not signed in, usage limit reached
            await self.send({"type": "reply_error", "id": rid, "error": f"Claude Code: {e}"})
            return False
        await self.send({"type": "reply_end", "id": rid, "skipped": result is None})
        return result is not None

    async def command(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "answer_now" and self.transcript:
            # Answer the last ~2 utterances: questions often span a segment boundary.
            await self._reply(" ".join(t for _, t in list(self.transcript)[-2:]), forced=True)
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
        for ts, text in agent.transcript:
            await socket.send_json({"type": "transcript", "text": text, "ts": ts})
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
