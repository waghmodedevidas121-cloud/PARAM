"""
Generation orchestration — Colab port of Voicebox services/generation.py.

Pipeline:
  load model → build voice prompt → generate_chunked → normalize → effects → save
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import config, history, profiles
from ..audio import has_tts_runaway, normalize_audio, save_audio, trim_tts_output
from ..backends import engine_needs_trim, get_model_config, get_tts_backend_for_engine
from ..chunked_tts import generate_chunked
from ..effects import apply_effects
from .model_manager import get_model_manager


def _prepare_voice_prompt(backend, profile: dict, engine: str) -> dict:
    voice_type = profile.get("voice_type") or "cloned"
    if voice_type == "preset":
        return profiles.voice_prompt_for_profile(profile)
    prompt = profiles.voice_prompt_for_profile(profile)
    # Qwen / LuxTTS / TADA encode the reference up front.
    # Chatterbox stores the path and reads it at generate() time.
    if engine in ("qwen", "luxtts", "tada"):
        return backend.create_voice_prompt(prompt["ref_audio"], prompt.get("ref_text") or "")
    return prompt


def run_generation(
    *,
    text: str,
    model_name: str,
    profile: dict,
    language: str = "en",
    seed: Optional[int] = None,
    instruct: Optional[str] = None,
    exaggeration: Optional[float] = None,
    max_chunk_chars: int = 800,
    crossfade_ms: int = 50,
    normalize: bool = True,
    effects_chain: Optional[list] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required.")
    if len(text) > 50000:
        raise ValueError("Text exceeds Voicebox's 50,000 character limit.")

    cfg = get_model_config(model_name)
    if not cfg:
        raise ValueError(f"Unknown model: {model_name}")

    manager = get_model_manager()
    if progress:
        progress(f"Ensuring {cfg.display_name} is loaded…")
    manager.ensure_loaded(model_name, progress=progress)

    backend = get_tts_backend_for_engine(cfg.engine)
    if progress:
        progress("Preparing voice prompt…")
    voice_prompt = _prepare_voice_prompt(backend, profile, cfg.engine)

    trim_fn = trim_tts_output if engine_needs_trim(cfg.engine) else None
    # runaway retry is an MLX-only Voicebox flag; Colab is PyTorch.
    runaway_detector = None

    def _one(chunk_text: str, chunk_seed):
        return backend.generate(
            chunk_text,
            voice_prompt,
            language=language,
            seed=chunk_seed,
            instruct=instruct,
            exaggeration=exaggeration,
        )

    audio, sample_rate = generate_chunked(
        _one,
        text,
        max_chunk_chars=max_chunk_chars,
        crossfade_ms=crossfade_ms,
        seed=seed,
        trim_fn=trim_fn,
        runaway_detector=runaway_detector,
        progress=progress,
    )

    if normalize:
        if progress:
            progress("Normalizing…")
        audio = normalize_audio(np.asarray(audio, dtype=np.float32))

    if effects_chain and any(e.get("enabled", True) for e in effects_chain):
        if progress:
            progress("Applying pedalboard effects…")
        audio = apply_effects(audio, sample_rate, effects_chain)

    generation_id = str(uuid.uuid4())
    out_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(out_path), sample_rate)
    duration = float(len(audio) / sample_rate) if sample_rate else 0.0

    if progress:
        progress(f"Done — {duration:.2f}s")

    return {
        "id": generation_id,
        "audio_path": str(out_path),
        "duration": duration,
        "sample_rate": sample_rate,
        "engine": cfg.engine,
        "model_name": cfg.model_name,
        "language": language,
    }


def record_history(result: dict, *, profile_name: str, expression: str, text: str) -> dict:
    return history.add_entry(
        profile_name=profile_name,
        engine=result["engine"],
        language=result["language"],
        expression=expression,
        text=text,
        audio_path=result["audio_path"],
        duration=result["duration"],
        model_name=result["model_name"],
    )
