"""Speech synthesis with emotional prosody.

Base voice: Piper (local ONNX neural TTS). Emotion is rendered by applying
signal-level prosody to the neutral synthesis:
  * pitch shift   - semitone transposition via resampling (phase-preserving)
  * duration      - time-stretch via WSOLA (keeps pitch independent of speed)
  * volume        - gain with soft limiter
  * tremolo       - slow amplitude modulation (sad / fear quiver)
  * vibrato       - periodic pitch wobble (irony, excitement)
"""
from __future__ import annotations

import io
import math
import os
import threading
import wave

import numpy as np
from scipy.signal import resample_poly

from .emotion import EmotionProfile

MODEL_PATH = os.environ.get(
    "PIPER_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "en_US-lessac-medium.onnx"))

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
    """
    if abs(semitones) < 0.05:
        return x
    ratio = semitones_to_ratio(semitones)
    tmp = resample_poly(x, round(sr * ratio * 100), sr * 100).astype(np.float32)
    n = int(len(tmp) / ratio)
    idx = np.linspace(0, len(tmp) - 1, n)
    out = np.interp(idx, np.arange(len(tmp)), tmp).astype(np.float32)
    # match original length exactly
    idx2 = np.linspace(0, len(out) - 1, len(x))
    return np.interp(idx2, np.arange(len(out)), out).astype(np.float32)


def time_stretch(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """WSOLA time stretching (vectorized). factor<1 speeds up, >1 slows down."""
    if abs(factor - 1.0) < 0.01 or len(x) == 0:
        return x
    hop_out = max(int(sr * 0.015), 1)   # 15 ms output hop
    win = max(int(sr * 0.040), 4)       # 40 ms window
    hop_in = max(int(hop_out * factor), 1)
    tol = int(sr * 0.010)               # OLA search tolerance

    n_out = int(len(x) / factor)
    # pad input so every read window is full-size; cap total memory (~64 MB)
    max_frames = min(n_out // hop_out + 1, (len(x) - win) // hop_in + 1)
    max_frames = max(max_frames, 0)
    need = max_frames * max(hop_in, hop_out) + win + 2 * tol + 8
    x = np.pad(x.astype(np.float32), (0, max(need - len(x), 0)))
    n_frames = min(n_out // hop_out + 1, (len(x) - win) // hop_in + 1,
                   (len(x) - win) // hop_out + 1)
    if n_frames <= 0:
        return x[:n_out] if n_out else x

    # candidate input positions per frame
    centers = np.arange(n_frames) * hop_in                       # [F]
    offsets = np.arange(-tol, tol + 1, dtype=np.int64)           # [S]
    cand = np.clip(centers[:, None] + offsets[None, :], 0, len(x) - win)  # [F,S]

    w = np.hanning(win).astype(np.float32)
    out_len = (n_frames - 1) * hop_out + win
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    prev_best = 0
    for f in range(n_frames):
        pos_out = f * hop_out
        target = x[pos_out:pos_out + win] * w
        if not target.any():
            best = int(min(centers[f], len(x) - win))
        else:
            c = cand[f]
            # restrict to monotonic candidates (>= previous choice)
            valid = c[c >= prev_best]
            if len(valid) == 0:
                valid = c
            segs = x[valid][:, None] * w[None, :]                # [V,win]
            scores = segs @ target
            best = int(valid[int(np.argmax(scores))])
        out[pos_out:pos_out + win] += x[best:best + win] * w
        norm[pos_out:pos_out + win] += w
        prev_best = best
    norm[norm < 1e-6] = 1e-6
    out /= norm
    return out[:n_out]


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


def limit(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > ceiling:
        x = x * (ceiling / peak)
    return x


def synthesize_emotional(text: str, prof: EmotionProfile) -> bytes:
    """Synthesize one passage with emotion -> WAV bytes."""
    x, sr = synth_plain(text)
    if len(x) == 0:
        x = silence(200, sr)

    if prof.vibrato > 0.02:
        x = apply_vibrato(x, sr, prof.vibrato * 2.2)
    if abs(prof.pitch) > 0.05:
        x = change_pitch(x, sr, prof.pitch)
    speed_factor = 1.0 / prof.rate  # rate>1 => faster => shorter
    x = time_stretch(x, sr, speed_factor)
    x = tremolo(x, sr, prof.tremolo)
    x = x * prof.volume
    x = limit(x)

    head = silence(prof.pause_before_ms, sr)
    tail = silence(180, sr)
    full = np.concatenate([head, x, tail])

    pcm = (np.clip(full, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
