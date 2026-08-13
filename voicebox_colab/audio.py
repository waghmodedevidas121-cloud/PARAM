"""
Audio utilities ported from Voicebox backend/utils/audio.py.

Kept behaviour:
  - RMS normalize + peak limit
  - load / atomic save
  - trim_tts_output (Chatterbox trailing-noise cut)
  - has_tts_runaway
  - reference-audio preprocess + validate (2–30 s)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
    peak_limit: float = 0.85,
) -> np.ndarray:
    audio = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    target_rms = 10 ** (target_db / 20)
    if rms > 0:
        audio = audio * (target_rms / rms)
    return np.clip(audio, -peak_limit, peak_limit)


def load_audio(
    path: str,
    sample_rate: int = 24000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    if librosa is not None:
        audio, sr = librosa.load(path, sr=sample_rate, mono=mono)
        return np.asarray(audio, dtype=np.float32), int(sr)
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1 and mono:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        if librosa is None:
            # Linear resample fallback so a missing librosa does not hard-fail.
            duration = len(audio) / float(sr)
            target_len = int(duration * sample_rate)
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
            sr = sample_rate
    return np.asarray(audio, dtype=np.float32), int(sr)


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int = 24000,
) -> None:
    temp_path = f"{path}.tmp"
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(temp_path, audio, sample_rate, format="WAV")
        os.replace(temp_path, path)
    except Exception as exc:
        try:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception:
            pass
        raise OSError(f"Failed to save audio to {path}: {exc}") from exc


def has_tts_runaway(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    max_internal_silence_ms: int = 2000,
) -> bool:
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return False
    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    seen_speech = False
    consecutive_silence = 0
    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        is_speech = float(np.sqrt(np.mean(frame**2))) >= threshold_linear
        if is_speech:
            if seen_speech and consecutive_silence >= max_silence_frames:
                return True
            seen_speech = True
            consecutive_silence = 0
        elif seen_speech:
            consecutive_silence += 1
    return False


def trim_tts_output(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    min_silence_ms: int = 200,
    max_internal_silence_ms: int = 1000,
    fade_ms: int = 30,
) -> np.ndarray:
    """Cut Chatterbox-style [speech][silence][hallucinated noise]."""
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return audio

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)
    rms = np.array(
        [np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)]
    )
    is_speech = rms >= threshold_linear

    first_speech = 0
    for i, spoken in enumerate(is_speech):
        if spoken:
            first_speech = max(0, i - 1)
            break

    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    consecutive_silence = 0
    cut_frame = n_frames
    for i in range(first_speech, n_frames):
        if is_speech[i]:
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            if consecutive_silence >= max_silence_frames:
                cut_frame = i - consecutive_silence + 1
                break

    min_silence_frames = int(min_silence_ms / frame_ms)
    end_frame = cut_frame
    while end_frame > first_speech and not is_speech[end_frame - 1]:
        end_frame -= 1
    end_frame = min(end_frame + min_silence_frames, cut_frame)

    start_sample = first_speech * frame_len
    end_sample = min(end_frame * frame_len, len(audio))
    trimmed = audio[start_sample:end_sample].copy()

    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and len(trimmed) > fade_samples:
        fade = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        trimmed[-fade_samples:] *= fade
    return trimmed


def preprocess_reference_audio(
    audio: np.ndarray,
    sample_rate: int,
    peak_target: float = 0.95,
    trim_top_db: float = 40.0,
    edge_padding_ms: int = 100,
) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    if librosa is not None:
        trimmed, _ = librosa.effects.trim(audio, top_db=trim_top_db)
        if 0 < trimmed.size < audio.size:
            pad_each = int(sample_rate * edge_padding_ms / 1000)
            headroom = (audio.size - trimmed.size) // 2
            pad = min(pad_each, max(headroom, 0))
            if pad > 0:
                trimmed = np.pad(trimmed, (pad, pad), mode="constant")
            audio = trimmed
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > peak_target and peak > 0:
        audio = audio * (peak_target / peak)
    return audio


def validate_and_load_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    min_rms: float = 0.01,
) -> Tuple[bool, Optional[str], Optional[np.ndarray], Optional[int]]:
    try:
        audio, sr = load_audio(audio_path)
        audio = preprocess_reference_audio(audio, sr)
        duration = len(audio) / sr
        if duration < min_duration:
            return False, f"Audio too short (minimum {min_duration} seconds)", None, None
        if duration > max_duration:
            return False, f"Audio too long (maximum {max_duration} seconds)", None, None
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < min_rms:
            return False, "Audio is too quiet or silent", None, None
        return True, None, audio, sr
    except Exception as exc:
        return False, f"Error validating audio: {exc}", None, None
