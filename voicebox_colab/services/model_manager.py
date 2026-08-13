"""
Model manager — Voicebox-inspired, Colab-aware.

Desktop Voicebox can keep multiple backends instantiated; Colab VRAM
cannot. This manager:

  * loads only the selected model
  * unloads every other TTS backend first
  * reports downloaded / loaded / available (import-probed)
  * recommends models from the official VRAM table
"""

from __future__ import annotations

import logging
from typing import Optional

from ..backends import (
    TTS_MODEL_CONFIGS,
    get_model_config,
    get_tts_backend_for_engine,
    probe_import,
)
from ..backends.base import is_model_cached
from ..system import empty_cuda_cache, probe_system

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self._loaded_name: Optional[str] = None
        self._status: str = "Ready"

    @property
    def loaded_name(self) -> Optional[str]:
        return self._loaded_name

    @property
    def status(self) -> str:
        return self._status

    def list_status(self) -> list[dict]:
        info = probe_system()
        rows = []
        for cfg in TTS_MODEL_CONFIGS:
            available, reason = probe_import(cfg.engine)
            try:
                backend = get_tts_backend_for_engine(cfg.engine)
                loaded = bool(backend.is_loaded())
                if cfg.engine in ("qwen", "qwen_custom_voice", "tada"):
                    loaded = loaded and getattr(backend, "_current_model_size", None) in (
                        cfg.model_size,
                        None,
                    )
                    if cfg.engine == "tada":
                        loaded = bool(backend.is_loaded()) and getattr(backend, "model_size", None) == cfg.model_size
                    if cfg.engine in ("qwen", "qwen_custom_voice"):
                        loaded = bool(backend.is_loaded()) and getattr(
                            backend, "_current_model_size", None
                        ) == cfg.model_size
                downloaded = False
                if hasattr(backend, "is_cached"):
                    try:
                        downloaded = bool(backend.is_cached(cfg.model_size))
                    except TypeError:
                        downloaded = bool(backend.is_cached())
                else:
                    downloaded = is_model_cached(cfg.hf_repo_id)
            except Exception:
                loaded = False
                downloaded = is_model_cached(cfg.hf_repo_id)

            fits = (not info.cuda_available) or (cfg.vram_gb + 0.5 <= info.vram_total_gb) or cfg.vram_gb <= 1.0
            if loaded:
                state = "loaded"
            elif not available:
                state = "unavailable"
            elif downloaded:
                state = "downloaded"
            else:
                state = "available"

            rows.append(
                {
                    "model_name": cfg.model_name,
                    "display_name": cfg.display_name,
                    "engine": cfg.engine,
                    "hf_repo_id": cfg.hf_repo_id,
                    "model_size": cfg.model_size,
                    "size_mb": cfg.size_mb,
                    "vram_gb": cfg.vram_gb,
                    "downloaded": downloaded,
                    "loaded": loaded,
                    "available": available,
                    "import_reason": reason,
                    "recommended": fits and available,
                    "state": state,
                }
            )
        return rows

    def unload_all(self) -> None:
        for cfg in TTS_MODEL_CONFIGS:
            try:
                backend = get_tts_backend_for_engine(cfg.engine)
                if backend.is_loaded():
                    backend.unload_model()
            except Exception as exc:
                logger.warning("Unload %s failed: %s", cfg.engine, exc)
        self._loaded_name = None
        empty_cuda_cache()

    def unload(self, model_name: Optional[str] = None) -> str:
        name = model_name or self._loaded_name
        if not name:
            empty_cuda_cache()
            return "No model loaded."
        cfg = get_model_config(name)
        if not cfg:
            return f"Unknown model: {name}"
        backend = get_tts_backend_for_engine(cfg.engine)
        if backend.is_loaded():
            backend.unload_model()
        if self._loaded_name == name:
            self._loaded_name = None
        empty_cuda_cache()
        self._status = "Ready"
        return f"Unloaded {cfg.display_name}."

    def load(self, model_name: str, progress=None) -> str:
        cfg = get_model_config(model_name)
        if not cfg:
            raise ValueError(f"Unknown model: {model_name}")
        available, reason = probe_import(cfg.engine)
        if not available:
            raise RuntimeError(
                f"{cfg.display_name} is not installed in this runtime ({reason}). "
                "Run the matching install cell in the notebook."
            )

        info = probe_system()
        if info.cuda_available and cfg.vram_gb + 0.4 > info.vram_total_gb:
            logger.warning(
                "%s wants ~%.1f GB VRAM; GPU has %.1f GB. Load may OOM.",
                cfg.display_name,
                cfg.vram_gb,
                info.vram_total_gb,
            )

        if progress:
            progress(f"Unloading other engines before loading {cfg.display_name}…")
        # Always unload every other engine first — Colab cannot hold two TTS models.
        for other in TTS_MODEL_CONFIGS:
            if other.engine == cfg.engine:
                continue
            try:
                backend = get_tts_backend_for_engine(other.engine)
                if backend.is_loaded():
                    backend.unload_model()
            except Exception:
                pass
        empty_cuda_cache()

        if progress:
            progress(f"Loading {cfg.display_name} ({cfg.hf_repo_id})…")
        self._status = f"Loading {cfg.display_name}"
        backend = get_tts_backend_for_engine(cfg.engine)
        if cfg.engine in ("qwen", "qwen_custom_voice", "tada"):
            backend.load_model(cfg.model_size)
        else:
            backend.load_model()
        self._loaded_name = cfg.model_name
        self._status = "Ready"
        if progress:
            progress(f"Loaded {cfg.display_name}")
        return f"Loaded {cfg.display_name} on {info.device}."

    def ensure_loaded(self, model_name: str, progress=None) -> None:
        if self._loaded_name == model_name:
            cfg = get_model_config(model_name)
            if cfg:
                backend = get_tts_backend_for_engine(cfg.engine)
                if backend.is_loaded():
                    return
        self.load(model_name, progress=progress)


_MANAGER: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ModelManager()
    return _MANAGER
