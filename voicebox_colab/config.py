"""
Data-directory configuration for the Colab adaptation.

Voicebox desktop uses an OS app-data directory (backend/config.py).
On Colab we default to /content/voicebox_colab as specified; locally
we fall back to ./data/voicebox_colab so the same code runs outside Colab.
"""

from __future__ import annotations

import os
from pathlib import Path

_COLAB_DEFAULT = Path("/content/voicebox_colab")
_LOCAL_DEFAULT = Path("data/voicebox_colab").resolve()


def _default_data_dir() -> Path:
    override = os.environ.get("VOICEBOX_COLAB_DATA_DIR")
    if override:
        return Path(override).resolve()
    if _COLAB_DEFAULT.parent.exists() and (
        os.path.exists("/content") or os.environ.get("COLAB_RELEASE_TAG")
    ):
        return _COLAB_DEFAULT
    return _LOCAL_DEFAULT


_data_dir = _default_data_dir()


def set_data_dir(path: str | Path) -> Path:
    global _data_dir
    _data_dir = Path(path).resolve()
    _data_dir.mkdir(parents=True, exist_ok=True)
    return _data_dir


def get_data_dir() -> Path:
    _data_dir.mkdir(parents=True, exist_ok=True)
    return _data_dir


def get_profiles_dir() -> Path:
    path = get_data_dir() / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_generations_dir() -> Path:
    path = get_data_dir() / "generations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir() -> Path:
    path = get_data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_history_path() -> Path:
    return get_data_dir() / "history.json"


def apply_colab_env() -> None:
    """Point HuggingFace caches at the Colab data dir when running on Colab."""
    data = get_data_dir()
    models_dir = os.environ.get("VOICEBOX_MODELS_DIR")
    if models_dir:
        os.environ.setdefault("HF_HUB_CACHE", models_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", models_dir)
        return
    hf_cache = data / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
