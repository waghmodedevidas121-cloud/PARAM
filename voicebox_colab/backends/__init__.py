"""
TTS engine registry — ported from Voicebox backend/backends/__init__.py.

ModelConfig fields, HF repo IDs, sizes, languages, needs_trim, and
supports_instruct are copied from current main. Colab always uses the
PyTorch repos (no MLX).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from ..capabilities import ENGINE_LANGUAGES


@dataclass
class ModelConfig:
    """Declarative config for a downloadable model variant. Voicebox ModelConfig."""

    model_name: str
    display_name: str
    engine: str
    hf_repo_id: str
    model_size: str = "default"
    size_mb: int = 0
    vram_gb: float = 0.0
    needs_trim: bool = False
    retries_runaway: bool = False
    supports_instruct: bool = False
    languages: list[str] = field(default_factory=lambda: ["en"])


# Official VRAM column from docs/content/docs/developer/model-management.mdx
TTS_MODEL_CONFIGS: list[ModelConfig] = [
    ModelConfig(
        model_name="qwen-tts-1.7B",
        display_name="Qwen TTS 1.7B",
        engine="qwen",
        hf_repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        model_size="1.7B",
        size_mb=3500,
        vram_gb=6.0,
        supports_instruct=False,
        languages=list(ENGINE_LANGUAGES["qwen"]),
    ),
    ModelConfig(
        model_name="qwen-tts-0.6B",
        display_name="Qwen TTS 0.6B",
        engine="qwen",
        hf_repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        model_size="0.6B",
        size_mb=1200,
        vram_gb=2.0,
        supports_instruct=False,
        languages=list(ENGINE_LANGUAGES["qwen"]),
    ),
    ModelConfig(
        model_name="qwen-custom-voice-1.7B",
        display_name="Qwen CustomVoice 1.7B",
        engine="qwen_custom_voice",
        hf_repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        model_size="1.7B",
        size_mb=3500,
        vram_gb=6.0,
        supports_instruct=True,
        languages=list(ENGINE_LANGUAGES["qwen_custom_voice"]),
    ),
    ModelConfig(
        model_name="qwen-custom-voice-0.6B",
        display_name="Qwen CustomVoice 0.6B",
        engine="qwen_custom_voice",
        hf_repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        model_size="0.6B",
        size_mb=1200,
        vram_gb=2.0,
        supports_instruct=True,
        languages=list(ENGINE_LANGUAGES["qwen_custom_voice"]),
    ),
    ModelConfig(
        model_name="luxtts",
        display_name="LuxTTS (Fast, CPU-friendly)",
        engine="luxtts",
        hf_repo_id="YatharthS/LuxTTS",
        size_mb=300,
        vram_gb=1.0,
        languages=["en"],
    ),
    ModelConfig(
        model_name="chatterbox-tts",
        display_name="Chatterbox TTS (Multilingual)",
        engine="chatterbox",
        hf_repo_id="ResembleAI/chatterbox",
        size_mb=3200,
        vram_gb=3.0,
        needs_trim=True,
        languages=list(ENGINE_LANGUAGES["chatterbox"]),
    ),
    ModelConfig(
        model_name="chatterbox-turbo",
        display_name="Chatterbox Turbo (English, Tags)",
        engine="chatterbox_turbo",
        hf_repo_id="ResembleAI/chatterbox-turbo",
        size_mb=1500,
        vram_gb=1.5,
        needs_trim=True,
        languages=["en"],
    ),
    ModelConfig(
        model_name="tada-1b",
        display_name="TADA 1B (English)",
        engine="tada",
        hf_repo_id="HumeAI/tada-1b",
        model_size="1B",
        size_mb=4000,
        vram_gb=4.0,
        languages=["en"],
    ),
    ModelConfig(
        model_name="tada-3b-ml",
        display_name="TADA 3B Multilingual",
        engine="tada",
        hf_repo_id="HumeAI/tada-3b-ml",
        model_size="3B",
        size_mb=8000,
        vram_gb=8.0,
        languages=list(ENGINE_LANGUAGES["tada"]),
    ),
    ModelConfig(
        model_name="kokoro",
        display_name="Kokoro 82M",
        engine="kokoro",
        hf_repo_id="hexgrad/Kokoro-82M",
        size_mb=350,
        vram_gb=0.15,
        languages=list(ENGINE_LANGUAGES["kokoro"]),
    ),
]


TTS_ENGINES = {
    "qwen": "Qwen TTS",
    "qwen_custom_voice": "Qwen CustomVoice",
    "luxtts": "LuxTTS",
    "chatterbox": "Chatterbox TTS",
    "chatterbox_turbo": "Chatterbox Turbo",
    "tada": "TADA",
    "kokoro": "Kokoro",
}

ENGINE_OPTIONS: list[dict[str, str]] = [
    {"value": "qwen:1.7B", "label": "Qwen3-TTS 1.7B", "engine": "qwen", "model_name": "qwen-tts-1.7B"},
    {"value": "qwen:0.6B", "label": "Qwen3-TTS 0.6B", "engine": "qwen", "model_name": "qwen-tts-0.6B"},
    {
        "value": "qwen_custom_voice:1.7B",
        "label": "Qwen CustomVoice 1.7B",
        "engine": "qwen_custom_voice",
        "model_name": "qwen-custom-voice-1.7B",
    },
    {
        "value": "qwen_custom_voice:0.6B",
        "label": "Qwen CustomVoice 0.6B",
        "engine": "qwen_custom_voice",
        "model_name": "qwen-custom-voice-0.6B",
    },
    {"value": "luxtts", "label": "LuxTTS", "engine": "luxtts", "model_name": "luxtts"},
    {"value": "chatterbox", "label": "Chatterbox", "engine": "chatterbox", "model_name": "chatterbox-tts"},
    {
        "value": "chatterbox_turbo",
        "label": "Chatterbox Turbo",
        "engine": "chatterbox_turbo",
        "model_name": "chatterbox-turbo",
    },
    {"value": "tada:1B", "label": "TADA 1B", "engine": "tada", "model_name": "tada-1b"},
    {"value": "tada:3B", "label": "TADA 3B Multilingual", "engine": "tada", "model_name": "tada-3b-ml"},
    {"value": "kokoro", "label": "Kokoro 82M", "engine": "kokoro", "model_name": "kokoro"},
]

ENGINE_DESCRIPTIONS = {
    "qwen": "Multi-language, two sizes",
    "qwen_custom_voice": "9 preset voices, instruct control",
    "luxtts": "Fast, English-focused",
    "chatterbox": "23 languages, incl. Hebrew",
    "chatterbox_turbo": "English, [laugh] [cough] tags",
    "tada": "HumeAI, 700s+ coherent audio",
    "kokoro": "82M params, CPU realtime, 8 langs",
}


def get_tts_model_configs() -> list[ModelConfig]:
    return list(TTS_MODEL_CONFIGS)


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    for cfg in TTS_MODEL_CONFIGS:
        if cfg.model_name == model_name:
            return cfg
    return None


def parse_engine_selection(value: str) -> tuple[str, str, str]:
    """Return (engine, model_size, model_name) from an ENGINE_OPTIONS value."""
    for opt in ENGINE_OPTIONS:
        if opt["value"] == value:
            cfg = get_model_config(opt["model_name"])
            size = cfg.model_size if cfg else "default"
            return opt["engine"], size, opt["model_name"]
    if ":" in value:
        engine, size = value.split(":", 1)
        for cfg in TTS_MODEL_CONFIGS:
            if cfg.engine == engine and cfg.model_size == size:
                return engine, size, cfg.model_name
        return engine, size, value
    for cfg in TTS_MODEL_CONFIGS:
        if cfg.engine == value:
            return value, cfg.model_size, cfg.model_name
    raise ValueError(f"Unknown engine selection: {value}")


def engine_needs_trim(engine: str) -> bool:
    for cfg in TTS_MODEL_CONFIGS:
        if cfg.engine == engine:
            return cfg.needs_trim
    return False


_backends: dict[str, object] = {}
_lock = threading.Lock()


def get_tts_backend_for_engine(engine: str):
    """Factory — same dispatch as Voicebox get_tts_backend_for_engine()."""
    if engine in _backends:
        return _backends[engine]
    with _lock:
        if engine in _backends:
            return _backends[engine]
        if engine == "qwen":
            from .qwen import QwenTTSBackend

            backend = QwenTTSBackend()
        elif engine == "qwen_custom_voice":
            from .qwen_custom_voice import QwenCustomVoiceBackend

            backend = QwenCustomVoiceBackend()
        elif engine == "luxtts":
            from .luxtts import LuxTTSBackend

            backend = LuxTTSBackend()
        elif engine == "chatterbox":
            from .chatterbox import ChatterboxTTSBackend

            backend = ChatterboxTTSBackend()
        elif engine == "chatterbox_turbo":
            from .chatterbox_turbo import ChatterboxTurboTTSBackend

            backend = ChatterboxTurboTTSBackend()
        elif engine == "tada":
            from .tada import HumeTadaBackend

            backend = HumeTadaBackend()
        elif engine == "kokoro":
            from .kokoro import KokoroTTSBackend

            backend = KokoroTTSBackend()
        else:
            raise ValueError(f"Unknown TTS engine: {engine}. Supported: {list(TTS_ENGINES)}")
        _backends[engine] = backend
        return backend


def reset_backends() -> None:
    _backends.clear()


def probe_import(engine: str) -> tuple[bool, str]:
    """Return (available, reason) without loading weights."""
    try:
        if engine == "qwen":
            import qwen_tts  # noqa: F401

            return True, "qwen-tts installed"
        if engine == "qwen_custom_voice":
            import qwen_tts  # noqa: F401

            return True, "qwen-tts installed"
        if engine == "luxtts":
            from zipvoice.luxvoice import LuxTTS  # noqa: F401

            return True, "LuxTTS installed"
        if engine == "chatterbox":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: F401

            return True, "chatterbox-tts installed"
        if engine == "chatterbox_turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: F401

            return True, "chatterbox-tts (turbo) installed"
        if engine == "tada":
            import tada  # noqa: F401

            return True, "hume-tada installed"
        if engine == "kokoro":
            import kokoro  # noqa: F401

            return True, "kokoro installed"
        return False, f"unknown engine {engine}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
