"""Emotional E-book Reader - FastAPI backend.

Upload a book (EPUB/FB2/TXT/HTML) -> it is split into passages, each passage
gets an emotion analysis and an emotionally-synthesized audio clip. The SPA
frontend streams playback and shows the current text with its emotion badge.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import books as books_mod
from .emotion import analyze, EMOTIONS
from .tts import synthesize_emotional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "60"))
MAX_PASSAGES = int(os.environ.get("MAX_PASSAGES", "400"))  # demo-friendly cap

ALLOWED_EXT = {".epub", ".fb2", ".txt", ".md", ".html", ".htm"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("emo-reader")

app = FastAPI(title="Эмоциональная читалка", version="1.0")


# ---------------------------------------------------------------------------
# In-memory registry (persisted per book folder via meta.json)
# ---------------------------------------------------------------------------
class BookState:
    def __init__(self, book: books_mod.Book):
        self.book = book
        self.dir = DATA_DIR / book.id
        self.meta_path = self.dir / "meta.json"
        self.status = "ready" if self.meta_path.exists() else "new"
        self.progress = 0.0
        self.error = ""
        self.analyzing = False
        if self.meta_path.exists():
            try:
                saved = json.loads(self.meta_path.read_text())
                self._restore(saved)
            except Exception:
                log.exception("bad meta for %s", book.id)

    def _restore(self, saved: dict):
        metas = saved.get("passages", [])
        for p, m in zip(self.book.passages, metas):
            p.emotion = m["emotion"]
            p.confidence = m.get("confidence", 0.5)
            p.reason = m.get("reason", "")
            p.audio = m.get("audio", f"{p.index}.wav")
            p.duration = m.get("duration", 0.0)
        self.status = saved.get("status", "ready")
        self.progress = 1.0

    def save_meta(self):
        metas = []
        for p in self.book.passages:
            metas.append({
                "emotion": getattr(p, "emotion", None),
                "confidence": getattr(p, "confidence", None),
                "reason": getattr(p, "reason", ""),
                "audio": getattr(p, "audio", None),
                "duration": getattr(p, "duration", 0.0),
            })
        self.meta_path.write_text(json.dumps(
            {"status": self.status, "passages": metas}, ensure_ascii=False))


REGISTRY: dict[str, BookState] = {}


def load_existing_books():
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "meta.json").exists() and (d / "source").exists():
            try:
                b = books_mod.load_book(str(d / "source"), MAX_PASSAGES)
                b.id = d.name
                st = BookState(b)
                REGISTRY[b.id] = st
            except Exception:
                log.exception("failed to load book dir %s", d)


def get_state(book_id: str) -> BookState:
    st = REGISTRY.get(book_id)
    if not st:
        raise HTTPException(404, "Книга не найдена")
    return st


def book_json(st: BookState) -> dict:
    b = st.book
    passages = []
    for p in b.passages:
        passages.append({
            "index": p.index,
            "text": p.text,
            "chapter": p.chapter,
            "emotion": getattr(p, "emotion", None),
            "confidence": getattr(p, "confidence", None),
            "reason": getattr(p, "reason", ""),
            "ready": getattr(p, "audio", None) is not None,
            "duration": getattr(p, "duration", 0.0),
        })
    return {
        "id": b.id, "title": b.title, "author": b.author, "language": b.language,
        "status": st.status, "progress": round(st.progress, 3), "error": st.error,
        "num_passages": len(b.passages),
        "emotions": [e for e in EMOTIONS],
        "passages": passages,
    }


# ---------------------------------------------------------------------------
# Processing pipeline (analyze + synthesize sequentially in a worker thread)
# ---------------------------------------------------------------------------
import io  # noqa: E402

from concurrent.futures import ThreadPoolExecutor  # noqa: E402

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")


def _process_book_safe(st: BookState):
    """Run synthesis; on any failure mark error but keep partial results.

    Memory guards: the whole pipeline runs in a single worker thread, one
    passage at a time, and transient buffers are freed after each passage so
    that long "sad" (slow + tremolo) fragments can't pile up memory and get
    the process OOM-killed (exit code 137).
    """
    try:
        _process_book_inner(st)
    except MemoryError as e:
        log.exception("processing ran out of memory")
        st.status = "error"
        st.error = ("Недостаточно памяти для озвучки. Уменьшите MAX_PASSAGES "
                    "или разбейте книгу на части.")
        st.save_meta()
    except Exception as e:
        log.exception("processing failed")
        st.status = "error"
        st.error = str(e)[:300]
        st.save_meta()
    finally:
        st.analyzing = False


def _process_book_inner(st: BookState):
    total = len(st.book.passages)
    prev_emo = None
    for p in st.book.passages:
        if getattr(p, "audio", None):          # already synthesized
            continue
        prof = analyze(p.text, prev_emo)
        prev_emo = prof.emotion
        wav_bytes = synthesize_emotional(p.text, prof)
        out_name = f"{p.index}.mp3"
        _wav_to_mp3(wav_bytes, st.dir / out_name)
        p.emotion = prof.emotion
        p.confidence = prof.confidence
        p.reason = prof.reason
        p.audio = out_name
        p.duration = _wav_duration(wav_bytes)
        del wav_bytes                          # free audio buffer promptly
        st.progress = (p.index + 1) / total
        if p.index % 5 == 0 or p.index == total - 1:
            st.save_meta()
            log.info("book %s progress %.0f%%", st.book.id, st.progress * 100)
    st.status = "ready"
    st.progress = 1.0
    st.save_meta()


def _wav_duration(wav_bytes: bytes) -> float:
    import wave
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _wav_to_mp3(wav_bytes: bytes, out_path: Path):
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    seg.export(str(out_path), format="mp3", bitrate="48k")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/books")
async def upload_book(file: UploadFile = File(...)):
    name = file.filename or "book"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400,
                            f"Формат {ext} не поддерживается. Загрузите EPUB, FB2, TXT, MD или HTML.")
    bid = uuid.uuid4().hex[:12]
    bdir = DATA_DIR / bid
    bdir.mkdir(parents=True)
    src = bdir / ("source" + ext)
    size = 0
    with open(src, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                shutil.rmtree(bdir, ignore_errors=True)
                raise HTTPException(413, f"Файл больше {MAX_UPLOAD_MB} МБ")
            f.write(chunk)

    try:
        book = await asyncio.to_thread(books_mod.load_book, str(src), MAX_PASSAGES)
    except ValueError as e:
        shutil.rmtree(bdir, ignore_errors=True)
        raise HTTPException(400, str(e))
    except Exception:
        log.exception("parse failed")
        shutil.rmtree(bdir, ignore_errors=True)
        raise HTTPException(400, "Не удалось разобрать файл книги")

    if not book.passages:
        shutil.rmtree(bdir, ignore_errors=True)
        raise HTTPException(400, "В книге не найдено текста для озвучки")

    book.id = bid                      # keep folder == id
    st = BookState(book)
    st.status = "processing"
    REGISTRY[bid] = st

    # Single shared worker thread: books are processed one at a time so that
    # concurrent uploads can't multiply memory usage (OOM guard).
    _EXECUTOR.submit(_process_book_safe, st)
    return book_json(st)


@app.get("/api/books")
def list_books():
    out = []
    for st in REGISTRY.values():
        j = book_json(st)
        j.pop("passages", None)
        out.append(j)
    return sorted(out, key=lambda x: x["title"])


@app.get("/api/books/{book_id}")
def get_book(book_id: str):
    return book_json(get_state(book_id))


@app.get("/api/books/{book_id}/audio/{index}")
def get_audio(book_id: str, index: int):
    st = get_state(book_id)
    if index < 0 or index >= len(st.book.passages):
        raise HTTPException(404, "Фрагмент не найден")
    p = st.book.passages[index]
    fname = getattr(p, "audio", None)
    if not fname:
        raise HTTPException(425, "Аудио ещё не готово")
    path = st.dir / fname
    if not path.exists():
        raise HTTPException(404, "Файл аудио потерян")
    return FileResponse(path, media_type="audio/mpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.delete("/api/books/{book_id}")
def delete_book(book_id: str):
    st = REGISTRY.pop(book_id, None)
    if not st:
        raise HTTPException(404, "Книга не найдена")
    shutil.rmtree(st.dir, ignore_errors=True)
    return {"ok": True}


@app.get("/api/emotions")
def emotions():
    return {"emotions": EMOTIONS}


@app.on_event("startup")
def _startup():
    load_existing_books()
    log.info("loaded %d existing books", len(REGISTRY))


# static frontend last so /api/* wins
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
