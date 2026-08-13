"""
Shared backend helpers — ported from Voicebox backend/backends/base.py.

Colab is CUDA-or-CPU (no MLX / DirectML / XPU path required).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..audio import load_audio, normalize_audio

logger = logging.getLogger(__name__)


def is_model_cached(
    hf_repo: str,
    *,
    weight_extensions: tuple[str, ...] = (".safetensors", ".bin", ".pt", ".pth"),
    required_files: Optional[list[str]] = None,
) -> bool:
    try:
        from huggingface_hub import constants as hf_constants

        repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + hf_repo.replace("/", "--"))
        if not repo_cache.exists():
            return False
        blobs_dir = repo_cache / "blobs"
        if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
            return False
        snapshots_dir = repo_cache / "snapshots"
        if not snapshots_dir.exists():
            return False
        if required_files:
            for fname in required_files:
                if not any(snapshots_dir.rglob(fname)):
                    return False
            return True
        for ext in weight_extensions:
            if any(snapshots_dir.rglob(f"*{ext}")):
                return True
        return False
    except Exception as exc:
        logger.warning("Error checking cache for %s: %s", hf_repo, exc)
        return False


def get_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def empty_device_cache(device: str) -> None:
    try:
        import gc

        gc.collect()
        import torch

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def manual_seed(seed: int, device: str) -> None:
    import torch

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def combine_voice_prompts(
    audio_paths: List[str],
    reference_texts: List[str],
    *,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    combined_audio = []
    for path in audio_paths:
        kwargs = {"sample_rate": sample_rate} if sample_rate else {}
        audio, _sr = load_audio(path, **kwargs)
        audio = normalize_audio(audio)
        combined_audio.append(audio)
    mixed = normalize_audio(np.concatenate(combined_audio))
    return mixed, " ".join(reference_texts)


def patch_chatterbox_f32(model) -> None:
    """
    librosa.load returns float64. Chatterbox matmuls those against float32
    weights. Voicebox patches the two known entry points.
    """
    import types

    _tokzr = model.s3gen.tokenizer
    _orig_log_mel = _tokzr.log_mel_spectrogram.__func__

    def _f32_log_mel(self_tokzr, audio, padding=0):
        import torch as _torch

        if _torch.is_tensor(audio):
            audio = audio.float()
        return _orig_log_mel(self_tokzr, audio, padding)

    _tokzr.log_mel_spectrogram = types.MethodType(_f32_log_mel, _tokzr)

    _ve = model.ve
    _orig_ve_forward = _ve.forward.__func__

    def _f32_ve_forward(self_ve, mels):
        return _orig_ve_forward(self_ve, mels.float())

    _ve.forward = types.MethodType(_f32_ve_forward, _ve)
