"""
Chunked TTS generation — ported from Voicebox backend/utils/chunked_tts.py.

Workflow (Voicebox current main):
  1. Split text at sentence boundaries (abbreviations, CJK, [tags] atomic)
  2. Generate each chunk independently
  3. Concatenate with crossfade (default 50 ms)
  4. Optional per-chunk trim (Chatterbox) and runaway retry (MLX Qwen)

Short text (≤ max_chunk_chars) uses the single-shot fast path.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("voicebox_colab.chunked_tts")

DEFAULT_MAX_CHUNK_CHARS = 800
MAX_RUNAWAY_RETRIES = 2
MIN_RUNAWAY_RETRY_CHARS = 100

_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
        "inc", "ltd", "corp", "dept", "est", "approx", "vs", "etc",
        "e.g", "i.e", "a.m", "p.m", "u.s", "u.s.a", "u.k",
    }
)

_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")

ProgressFn = Optional[Callable[[str], None]]


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        segment = remaining[:max_chars]
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            split_pos = _safe_hard_cut(segment, max_chars)
        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]
    return chunks


def _find_last_sentence_end(text: str) -> int:
    best = -1
    for match in re.finditer(r"[.!?](?:\s|$)", text):
        pos = match.start()
        char = text[pos]
        if char == ".":
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            if word_start >= 0 and text[word_start].isdigit():
                continue
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    for match in re.finditer(r"[\u3002\uff01\uff1f]", text):
        if match.start() > best:
            best = match.start()
    return best


def _find_last_clause_boundary(text: str) -> int:
    best = -1
    for match in re.finditer(r"[;:,\u2014](?:\s|$)", text):
        pos = match.start()
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    return best


def _inside_bracket_tag(text: str, pos: int) -> bool:
    for match in _PARA_TAG_RE.finditer(text):
        if match.start() < pos < match.end():
            return True
    return False


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    cut = max_chars - 1
    for match in _PARA_TAG_RE.finditer(segment):
        if match.start() < cut < match.end():
            return match.start() - 1 if match.start() > 0 else cut
    return cut


def concatenate_audio_chunks(
    chunks: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)
    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])
    return result


def generate_chunked(
    generate_fn,
    text: str,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    seed: Optional[int] = None,
    trim_fn=None,
    runaway_detector=None,
    progress: ProgressFn = None,
) -> Tuple[np.ndarray, int]:
    """
    Engine-agnostic chunked generation.

    ``generate_fn(chunk_text, chunk_seed) -> (audio, sample_rate)``
    is the only backend contract. Matches Voicebox TTSBackend.generate().
    """

    def generate_one(chunk_text: str, chunk_seed: Optional[int], retry_depth: int = 0):
        chunk_audio, chunk_sr = generate_fn(chunk_text, chunk_seed)
        if runaway_detector is not None and runaway_detector(chunk_audio, chunk_sr):
            if retry_depth >= MAX_RUNAWAY_RETRIES or len(chunk_text) <= MIN_RUNAWAY_RETRY_CHARS:
                raise RuntimeError("TTS output remained unstable after retrying smaller text chunks")
            retry_max_chars = max(MIN_RUNAWAY_RETRY_CHARS, len(chunk_text) // 2)
            retry_chunks = split_text_into_chunks(chunk_text, retry_max_chars)
            if len(retry_chunks) <= 1:
                raise RuntimeError("Unable to split unstable TTS output for retry")
            logger.warning(
                "Unstable TTS output for %d chars; retrying as %d smaller chunks",
                len(chunk_text),
                len(retry_chunks),
            )
            retry_audio = []
            sample_rate = chunk_sr
            for i, retry_text in enumerate(retry_chunks):
                retry_seed = (
                    chunk_seed + ((retry_depth + 1) * 1000) + i if chunk_seed is not None else None
                )
                audio, sample_rate = generate_one(retry_text, retry_seed, retry_depth + 1)
                retry_audio.append(np.asarray(audio, dtype=np.float32))
            return (
                concatenate_audio_chunks(retry_audio, sample_rate, crossfade_ms=crossfade_ms),
                sample_rate,
            )
        if trim_fn is not None:
            chunk_audio = trim_fn(chunk_audio, chunk_sr)
        return np.asarray(chunk_audio, dtype=np.float32), chunk_sr

    chunks = split_text_into_chunks(text, max_chunk_chars)
    if len(chunks) <= 1:
        if progress:
            progress("Generating (single shot)")
        return generate_one(text, seed)

    logger.info(
        "Splitting %d chars into %d chunks (max %d chars each)",
        len(text),
        len(chunks),
        max_chunk_chars,
    )
    audio_chunks: List[np.ndarray] = []
    sample_rate: Optional[int] = None
    for i, chunk_text in enumerate(chunks):
        msg = f"Generating chunk {i + 1} / {len(chunks)} ({len(chunk_text)} chars)"
        logger.info(msg)
        if progress:
            progress(msg)
        chunk_seed = (seed + i) if seed is not None else None
        chunk_audio, chunk_sr = generate_one(chunk_text, chunk_seed)
        audio_chunks.append(chunk_audio)
        if sample_rate is None:
            sample_rate = chunk_sr
    if progress:
        progress(f"Crossfading {len(audio_chunks)} chunks ({crossfade_ms} ms)")
    audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms)
    return audio, sample_rate
