"""Speech synthesis with emotional prosody.

Base voice: Piper (local ONNX neural TTS). Emotion is rendered by applying
signal-level prosody to the neutral synthesis:
  * pitch shift   - semitone transposition via resampling (phase-preserving)
  * duration      - time-stretch via WSOLA (keeps pitch independent of speed)
  * volume        - gain with soft limiter
  * tremolo       - slow amplitude modulation (sad / fear quiver)
  * vibrato       - periodic pitch wobble (irony, excitement)

Memory safety: long passages are synthesized sentence-by-sentence and every
DSP pass processes at most `_CHUNK` samples at a time. Without this, one very
long paragraph could allocate gigabytes in resample/WSOLA and OOM-kill the
process (exit code 137).
"""
from __future__ import annotations

import io
import logging
import math
import os
import re
import threading
import wave

import numpy as np
from scipy.signal import resample_poly

from .emotion import EmotionProfile

log = logging.getLogger("emo-reader.tts")

MODEL_PATH = os.environ.get(
    "PIPER_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "en_US-lessac-medium.onnx"))

# Max samples processed per DSP block (~20 s @ 22 kHz, float32 ≈ 1.8 MB/block)
_CHUNK = int(22050 * 20.0)
# Safety cap on passage length fed to the synthesizer in one call
_MAX_SENT_CHARS = 900

_voice = None
_lock = threading.Lock()


def get_voice():
    global _voice
    if _voice is None:
        with _lock:
            if _voice is None:
                from piper.voice import PiperVoice
                _voice = PiperVoice.load(MODEL_PATH)
    return _voice


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------

def semitones_to_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)


def change_pitch(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch shift without changing duration.

    Resample changes pitch AND length; we restore the length by simple
    linear resampling (a tiny length correction is imperceptible and avoids
    a second WSOLA pass on a longer signal).

    Memory-safe: scipy's polyphase filter is built in float64 with size
    ~2*max(up,down)*32 coefficients, so up/down must stay SMALL (<=100);
    otherwise one call allocates hundreds of MB and can OOM-kill the worker.
    """
    if abs(semitones) < 0.05:
        return x
    ratio = semitones_to_ratio(semitones)
    up, down = _resample_ratio(ratio)
    tmp = resample_poly(x, up, down).astype(np.float32)
    n = int(len(tmp) / ratio)
    idx = np.linspace(0, len(tmp) - 1, n)
    out = np.interp(idx, np.arange(len(tmp)), tmp).astype(np.float32)
    # match original length exactly
    idx2 = np.linspace(0, len(out) - 1, len(x))
    return np.interp(idx2, np.arange(len(out)), out).astype(np.float32)


def _resample_ratio(ratio: float) -> tuple[int, int]:
    """Approximate `ratio` as up/down with small ints (memory-bounded filters)."""
    from fractions import Fraction
    f = Fraction(float(ratio)).limit_denominator(97)
    return max(int(f.numerator), 1), max(int(f.denominator), 1)


def time_stretch(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """WSOLA time stretching. factor<1 speeds up, >1 slows down.

    Implemented frame-by-frame with a bounded search so peak memory stays at
    the size of the signal itself (the previous vectorized [F,S] / [V,win]
    candidate matrices allocated hundreds of MB per long passage and could
    OOM-kill the worker process).
    """
    if abs(factor - 1.0) < 0.01 or len(x) == 0:
        return x
    hop_out = max(int(sr * 0.015), 1)   # 15 ms output hop
    win = max(int(sr * 0.040), 4)       # 40 ms window
    hop_in = max(int(hop_out * factor), 1)
    tol = int(sr * 0.010)               # OLA search tolerance

    n_out = int(len(x) / factor)
    xi = x.astype(np.float32, copy=False)
    w = np.hanning(win).astype(np.float32)

    out_len = (n_out // hop_out + 2) * hop_out + win
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    last_start = 0                       # keep segment choice monotonic
    f = 0
    pos_out = 0
    while pos_out < n_out:
        center = min(f * hop_in, len(xi) - win)
        lo = max(last_start, center - tol)
        hi = min(center + tol, len(xi) - win)
        if hi < lo:
            hi = lo = max(0, min(center, len(xi) - win))
        target = xi[pos_out:pos_out + win]
        tl = len(target)
        if tl < win or not target.any():
            best = lo
        else:
            tw = target[:win] * w
            best, best_score = lo, -np.inf
            for p in range(lo, hi + 1):
                seg = xi[p:p + win]
                score = float(np.dot(seg, tw))
                if score > best_score:
                    best_score, best = score, p
        out[pos_out:pos_out + win] += xi[best:best + win] * w
        norm[pos_out:pos_out + win] += w
        last_start = best
        f += 1
        pos_out = f * hop_out
    norm[norm < 1e-6] = 1e-6
    out /= norm
    return out[:n_out]


def _map_chunks(func, x: np.ndarray, sr: int) -> np.ndarray:
    """Apply a *length-preserving* DSP function chunk by chunk (memory-safe)."""
    if len(x) <= _CHUNK:
        return func(x, sr)
    parts = []
    for i in range(0, len(x), _CHUNK):
        parts.append(func(x[i:i + _CHUNK], sr))
    return np.concatenate(parts)


def apply_vibrato(x: np.ndarray, sr: int, depth_semitones: float,
                  freq: float = 5.2) -> np.ndarray:
    """Periodic pitch wobble implemented as time-varying resampling."""
    if depth_semitones <= 0.02 or len(x) == 0:
        return x
    n = len(x)
    t = np.arange(n) / sr
    # instantaneous stretch factor oscillates around 1
    inst = 1.0 + (depth_semitones / 12.0) * np.sin(2 * np.pi * freq * t)
    # interpolate x at warped positions
    src_pos = np.cumsum(inst)
    src_pos = src_pos / src_pos[-1] * (n - 1)
    out = np.interp(np.arange(n), src_pos, x).astype(np.float32)
    return out


def tremolo(x: np.ndarray, sr: int, depth: float, freq: float = 4.5) -> np.ndarray:
    if depth <= 0.01 or len(x) == 0:
        return x
    t = np.arange(len(x)) / sr
    mod = 1.0 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * freq * t))
    return (x * mod).astype(np.float32)


def silence(ms: int, sr: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000.0), dtype=np.float32)


def synth_plain(text: str) -> tuple[np.ndarray, int]:
    """Neutral piper synthesis -> float32 mono samples."""
    voice = get_voice()
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.zeros(1, dtype=np.float32), 22050
    sr = chunks[0].sample_rate
    data = b"".join(c.audio_int16_bytes for c in chunks)
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized pieces bounded by _MAX_SENT_CHARS."""
    out: list[str] = []
    for s in _SENT_SPLIT.split(text.strip()):
        s = s.strip()
        if not s:
            continue
        while len(s) > _MAX_SENT_CHARS:          # pathological mega-sentence
            cut = s.rfind(",", 0, _MAX_SENT_CHARS)
            if cut < 60:
                cut = _MAX_SENT_CHARS
            out.append(s[:cut + 1])
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out or ([text.strip()] if text.strip() else [])


def limit(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > ceiling:
        x = x * (ceiling / peak)
    return x


def synthesize_emotional(text: str, prof: EmotionProfile) -> bytes:
    """Synthesize one passage with emotion -> WAV bytes.

    The passage is synthesized sentence by sentence and each sentence's audio
    is processed in bounded chunks, so peak memory stays small even for very
    long or very "sad" (slow, tremolo-heavy) fragments. This prevents the
    OOM-kill (exit code 137) that happened when a whole long paragraph was
    pushed through resample/WSOLA at once.
    """
    sr_ref = None
    parts: list[np.ndarray] = []
    for sent in _split_sentences(text):
        try:
            x, sr = synth_plain(sent)
        except MemoryError:
            log.warning("MemoryError synthesizing %r - skipping sentence", sent[:60])
            continue
        if len(x) == 0:
            continue
        sr_ref = sr
        if prof.vibrato > 0.02:
            x = _map_chunks(lambda a, r: apply_vibrato(a, r, prof.vibrato * 2.2), x, sr)
        if abs(prof.pitch) > 0.05:
            x = _map_chunks(lambda a, r: change_pitch(a, r, prof.pitch), x, sr)
        speed_factor = 1.0 / prof.rate  # rate>1 => faster => shorter
        x = time_stretch(x, sr, speed_factor)
        x = _map_chunks(lambda a, r: tremolo(a, r, prof.tremolo), x, sr)
        parts.append(x)
        del x

    sr = sr_ref or 22050
    if parts:
        full = np.concatenate(parts)
        del parts
        full = full * prof.volume
        full = limit(full)
    else:
        full = silence(400, sr)

    head = silence(prof.pause_before_ms, sr)
    tail = silence(180, sr)
    full = np.concatenate([head, full, tail])

    pcm = (np.clip(full, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
