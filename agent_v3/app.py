"""Echo (Live Reply Agent) - VERSION 3: Parakeet v3 speech-to-text (+ jargon fixer), fast replies. Listens to system audio, transcribes it, and drafts replies to
questions it hears. Also hears your microphone (labelled "You") so replies know what you've
already said. Open http://localhost:8766 for the live panel."""

import argparse
import asyncio
import difflib
import itertools
import os
import re
import subprocess
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from .audio import FileSource, MicSource, Segmenter, SystemAudioSource  # noqa: E402
from .responder import DOC_SUFFIXES, DOCS_DIR, Responder, looks_like_question  # noqa: E402
from .transcribe import Transcriber  # noqa: E402
from .vocab import build_vocabulary  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
TRANSCRIPT_WINDOW_S = 180  # how much recent conversation Claude sees
DUPLICATE_WINDOW_S = 30
TURN_GAP_S = 3.0  # lines separated by less silence than this are one speaker turn
INCOMPLETE_HOLD_S = 2.5  # wait this long for the rest of a half-finished sentence
SETTLE_S = 0.4
QUESTION_SETTLE_S = 0.15  # a line ending in "?" is very likely complete
MAX_WAIT_S = 12
QUESTION_MAX_WAIT_S = 1.0  # a finished "?" line is answered even if the next speaker is already talking
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ECHO_WINDOW_S = 10  # a mic line matching system audio heard this recently is the speakers, not you
ECHO_SIMILARITY = 0.6

TRAILING_FILLERS = {
    "the", "a", "an", "about", "of", "to", "for", "with", "and", "or", "but", "in", "on", "at",
    "your", "my", "our", "is", "are", "was", "what", "how", "so", "like", "that", "which", "um", "uh",
}


def is_bare_topic(text: str) -> bool:
    """A typed name with nothing else ("fibonacci", "binary search", "two sum"), which is
    answered with a Python solution and a short description."""
    words = text.split()
    return 0 < len(words) <= 4 and not re.search(r"[?.!,:;]", text) and not looks_like_question(text, "")


def settle_time(line: str) -> float:
    """How long to wait for more speech before answering."""
    if looks_incomplete(line):
        return INCOMPLETE_HOLD_S
    return QUESTION_SETTLE_S if line.rstrip().endswith("?") else SETTLE_S


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
        self.transcript: deque[tuple[float, str, str]] = deque()  # (end, text, speaker: them | you)
        self.recent_answers: dict[str, float] = {}
        self.ids = itertools.count(1)
        self.turn: list[str] = []  # lines of the current speaker turn, not yet answered
        self.turn_end = 0.0
        self.pending: asyncio.Task | None = None
        self.locked: asyncio.Task | None = None  # an in-flight answer to a complete question
        self.my_turn: list[str] = []  # the same, for what you say into your mic
        self.my_turn_end = 0.0
        self.my_pending: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

        self.responder = Responder(self.my_name)
        self.transcriber = Transcriber(
            os.getenv("WHISPER_MODEL", "base.en"), os.getenv("LANGUAGE") or None, self._on_text
        )
        self._update_vocabulary()
        pause_s = float(os.getenv("PAUSE_SECONDS", "0.5"))
        self.segmenter = segmenter = Segmenter(self.transcriber.submit, silence_s=pause_s)
        self.source = FileSource(segmenter, simulate) if simulate else SystemAudioSource(segmenter)
        # Your own voice gets its own segmenter, so it never merges with what others say.
        self.mic: MicSource | None = None
        self.mic_on = False
        if not simulate:
            self.mic = MicSource(Segmenter(self._on_mic_audio, silence_s=pause_s))
            self.mic_on = os.getenv("MIC", "on") != "off"

    def _on_mic_audio(self, audio) -> None:
        print(f"[mic] voice detected ({len(audio) / 16_000:.1f}s), transcribing", flush=True)
        self.transcriber.submit(audio, "you")

    def _update_vocabulary(self) -> None:
        terms = build_vocabulary(self.responder.docs + "\n" + self.responder.profile)
        self.transcriber.set_terms(terms)

    # --- plumbing -------------------------------------------------------------
    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.source.start()
        if self.mic and self.mic_on:
            self.mic.start()
        self.loop.create_task(self._meter())
        self.loop.create_task(self._keep_warm())

    async def _keep_warm(self) -> None:
        # Keeps a booted claude process ready for the current mode (and recycles old ones).
        while True:
            try:
                await self.responder.prewarm(self.mode)
                await self.responder.refresh_cache(self.mode)
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
            "version": "v3 parakeet",
            "mic": self._mic_state(),
        }

    def _mic_state(self) -> str | None:
        if not self.mic:
            return None
        if self.mic.failed:
            return "unavailable"
        return "on" if self.mic_on else "off"

    async def _meter(self) -> None:
        while True:
            mic_level = self.mic.level if self.mic and self.mic_on else 0.0
            await self.send({"type": "level", "value": max(self.source.level, mic_level)})
            await asyncio.sleep(0.2)

    # --- pipeline -------------------------------------------------------------
    def _on_text(self, text: str, start: float, end: float, speaker: str) -> None:
        """Called from the transcriber thread."""
        assert self.loop
        asyncio.run_coroutine_threadsafe(self._handle_text(text, start, end, speaker), self.loop)

    async def _is_echo(self, text: str, start: float) -> bool:
        """Without headphones the mic also picks up the call through the speakers. Waits for
        the system-audio copy of the same moment, then compares the two."""
        deadline = time.time() + 3
        while (self.segmenter.in_speech or self.transcriber.busy_with("them")) and time.time() < deadline:
            await asyncio.sleep(0.1)
        mine = text.lower().strip(".!?, ")
        for end, heard, speaker in reversed(self.transcript):
            if end < start - ECHO_WINDOW_S:
                break
            heard = heard.lower().strip(".!?, ")
            if speaker == "them" and (
                mine in heard or difflib.SequenceMatcher(None, mine, heard).ratio() >= ECHO_SIMILARITY
            ):
                return True
        return False

    def _labelled(self) -> list[str]:
        """Transcript lines for Claude; speaker labels only once your mic is part of it."""
        if not any(speaker == "you" for _, _, speaker in self.transcript):
            return [t for _, t, _ in self.transcript]
        me = f"{self.my_name or 'Me'} (the user)"
        return [f"{me if speaker == 'you' else 'Other'}: {t}" for _, t, speaker in self.transcript]

    async def _handle_text(self, text: str, start: float, end: float, speaker: str) -> None:
        if speaker == "you":
            if await self._is_echo(text, start):
                print(f"[mic] dropped as speaker echo: {text}", flush=True)
                return
            print(f"[mic] heard: {text}", flush=True)
        self.transcript.append((end, text, speaker))
        while self.transcript and self.transcript[0][0] < time.time() - TRANSCRIPT_WINDOW_S:
            self.transcript.popleft()
        await self.send({"type": "transcript", "text": text, "ts": start, "speaker": speaker})

        if speaker == "you":
            # Your own voice is answered too, as its own turn so it never merges with theirs.
            if start - self.my_turn_end > TURN_GAP_S:
                self.my_turn = []
            self.my_turn.append(text)
            self.my_turn_end = end
            if self.my_pending and not self.my_pending.done():
                self.my_pending.cancel()
            if self.auto != "off" and not self.source.paused:
                self.my_pending = asyncio.create_task(self._answer_my_turn())
            return

        # People often pause mid-question ("can you tell me about the..." ... "testing in
        # OnlineCook"), so consecutive lines form one turn that is answered as a whole.
        # A complete question that is already being answered stands: the next line (often
        # its answer) starts a new turn instead of cancelling and restarting the reply.
        locked = self.pending is not None and self.pending is self.locked and not self.pending.done()
        if locked or start - self.turn_end > TURN_GAP_S:
            self.turn = []
        self.turn.append(text)
        self.turn_end = end

        # Newer speech supersedes a queued or in-flight auto reply for the same turn.
        if self.pending and not self.pending.done() and not locked:
            self.pending.cancel()
        if self.auto != "off" and not self.source.paused:
            self.pending = asyncio.create_task(self._answer_turn())

    async def _answer_turn(self) -> None:
        turn = self.turn
        # Speech recognition often drops the "?", so question words count too.
        question = looks_like_question(turn[-1], self.my_name)
        await asyncio.sleep(settle_time(turn[-1]))
        # Don't answer while they're still talking or the next line is still being transcribed.
        # A finished question only waits briefly: in videos and busy calls nobody pauses.
        deadline = time.time() + (QUESTION_MAX_WAIT_S if question else MAX_WAIT_S)
        while (self.segmenter.in_speech or self.transcriber.busy) and time.time() < deadline:
            await asyncio.sleep(0.1)
        if self.turn is not turn:
            return  # a new line arrived and took over

        focus = " ".join(self.turn[-4:])
        key = focus.lower().strip()
        if time.time() - self.recent_answers.get(key, 0) < DUPLICATE_WINDOW_S:
            return
        self.recent_answers[key] = time.time()
        if question:
            self.locked = asyncio.current_task()
        # A detected question is always answered: left to judge, Claude skipped questions one
        # speaker asks another ("Sure. What is it?") and the card vanished without a trace.
        # Other lines still go to Claude, unforced, so a question the transcript garbled
        # beyond recognition is answered too; Claude stays silent on plain statements.
        if await self._reply(focus, forced=question) and self.turn is turn:
            self.turn = []  # answered; a follow-up starts a fresh turn

    async def _answer_my_turn(self) -> None:
        await asyncio.sleep(settle_time(self.my_turn[-1]))
        deadline = time.time() + MAX_WAIT_S
        while self.mic and (self.mic.segmenter.in_speech or self.transcriber.busy_with("you")) and time.time() < deadline:
            await asyncio.sleep(0.1)

        focus = " ".join(self.my_turn[-4:])
        # Only questions, even in "Every line" mode: the mic also hears the room, and
        # every stray line would otherwise become a Claude call and a reply card.
        if not looks_like_question(focus, ""):
            return
        key = focus.lower().strip()
        if time.time() - self.recent_answers.get(key, 0) < DUPLICATE_WINDOW_S:
            return
        self.recent_answers[key] = time.time()
        if await self._reply(focus, forced=True, own=True):
            self.my_turn = []

    async def _reply(self, focus: str, forced: bool, code_topic: bool = False, own: bool = False) -> bool:
        """Returns True if a reply was written."""
        rid = next(self.ids)
        lines = self._labelled()

        async def on_delta(delta: str) -> None:
            await self.send({"type": "reply_delta", "id": rid, "delta": delta})

        await self.send({"type": "reply_start", "id": rid, "question": focus, "forced": forced, "own": own})
        try:
            result = await self.responder.reply(lines, focus, self.mode, on_delta, forced=forced, code_topic=code_topic, own=own)
        except asyncio.CancelledError:
            await self.send({"type": "reply_cancel", "id": rid})
            raise
        except Exception as e:  # e.g. claude not signed in, usage limit reached
            await self.send({"type": "reply_error", "id": rid, "error": f"Claude Code: {e}"})
            return False
        await self.send({"type": "reply_end", "id": rid, "skipped": result is None})
        return result is not None

    def reload_docs(self) -> None:
        self.responder.reload()
        self._update_vocabulary()
        asyncio.create_task(self.responder.prewarm(self.mode))
        asyncio.create_task(self.responder.refresh_cache(self.mode))

    def doc_list(self) -> list[dict]:
        docs = []
        for name, text in self.responder.doc_texts.items():
            path = DOCS_DIR / name
            stat = path.stat() if path.exists() else None
            docs.append({
                "name": name,
                "words": len(text.split()),
                "bytes": stat.st_size if stat else 0,
                "modified": stat.st_mtime if stat else 0,
            })
        return docs

    async def command(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "answer_now" and self.transcript:
            # Answer the last ~2 utterances: questions often span a segment boundary.
            await self._reply(" ".join(t for _, t, _ in list(self.transcript)[-2:]), forced=True)
        elif kind == "ask" and msg.get("text", "").strip():
            text = msg["text"].strip()
            await self._reply(text, forced=True, code_topic=is_bare_topic(text))
        elif kind == "set_mode" and msg.get("mode") in ("meeting", "video"):
            self.mode = msg["mode"]
            asyncio.create_task(self.responder.prewarm(self.mode))
        elif kind == "set_auto" and msg.get("auto") in ("questions", "every", "off"):
            self.auto = msg["auto"]
        elif kind == "set_mic" and self.mic and msg.get("on") in (True, False):
            self.mic_on = msg["on"]
            print(f"[mic] turned {'on' if self.mic_on else 'off'} from the panel", flush=True)
            # Stop the helper rather than ignoring it, so macOS's mic indicator goes off too.
            self.mic.stop()
            if self.mic_on:
                self.mic.start()
        elif kind == "pause":
            self.source.paused = True
            if self.mic:
                self.mic.paused = True
        elif kind == "resume":
            self.source.paused = False
            if self.mic:
                self.mic.paused = False
        elif kind == "reload_docs":
            self.reload_docs()
        elif kind == "clear":
            self.transcript.clear()
        await self.send(self.state())


def build_app(agent: Agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        agent.start()
        yield
        agent.source.stop()
        if agent.mic:
            agent.mic.stop()
        agent.responder.drop_warm()

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/docs")
    async def list_docs() -> list[dict]:
        return agent.doc_list()

    @app.get("/api/docs/{name:path}")
    async def doc_text(name: str) -> dict:
        if name not in agent.responder.doc_texts:
            raise HTTPException(404, "No such doc")
        return {"name": name, "text": agent.responder.doc_texts[name]}

    @app.put("/api/docs/{filename}")
    async def upload_doc(filename: str, request: Request) -> dict:
        # The file is sent as the raw request body, so no multipart parser is needed.
        stem, suffix = Path(filename).stem, Path(filename).suffix.lower()
        if suffix not in DOC_SUFFIXES:
            raise HTTPException(415, f"Only {', '.join(DOC_SUFFIXES)} files are supported")
        stem = re.sub(r"[^\w\- .()]+", "_", stem).strip(" .") or "upload"
        body = await request.body()
        if not body:
            raise HTTPException(400, "Empty file")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than 25 MB")
        target = DOCS_DIR / f"{stem}{suffix}"
        n = 2
        while target.exists():  # never overwrite an existing doc
            target = DOCS_DIR / f"{stem} ({n}){suffix}"
            n += 1
        DOCS_DIR.mkdir(exist_ok=True)
        target.write_bytes(body)
        agent.reload_docs()
        await agent.send(agent.state())
        return {"saved": target.name}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        agent.clients.add(socket)
        await socket.send_json(agent.state())
        for ts, text, speaker in agent.transcript:
            await socket.send_json({"type": "transcript", "text": text, "ts": ts, "speaker": speaker})
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
    # 0.0.0.0 opens the panel to other devices on your network (no login, so trusted networks only).
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    args = parser.parse_args()

    agent = Agent(args.simulate)
    print(f"\n  Echo → http://localhost:{args.port}")
    if args.host != "127.0.0.1":
        lan_ip = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True).stdout.strip()
        print(f"  Other devices on your network → http://{lan_ip or '<this Mac IP>'}:{args.port}")
    print()
    uvicorn.run(build_app(agent), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
