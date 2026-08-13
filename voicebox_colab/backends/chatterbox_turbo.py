"""
Chatterbox Turbo backend — ported from Voicebox
backend/backends/chatterbox_turbo_backend.py.

English-only. Paralinguistic tags in the text are the expression system.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import ClassVar, Optional, Tuple

import numpy as np

from .base import (
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    manual_seed,
    patch_chatterbox_f32,
)

logger = logging.getLogger(__name__)

CHATTERBOX_TURBO_HF_REPO = "ResembleAI/chatterbox-turbo"
_TURBO_WEIGHT_FILES = [
    "t3_turbo_v1.safetensors",
    "s3gen_meanflow.safetensors",
    "ve.safetensors",
]


class ChatterboxTurboTTSBackend:
    _load_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def is_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(CHATTERBOX_TURBO_HF_REPO, required_files=_TURBO_WEIGHT_FILES)

    def load_model(self, model_size: str = "default") -> None:
        if self.model is not None:
            return
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        from huggingface_hub import snapshot_download

        device = get_torch_device()
        self._device = device
        logger.info("Loading Chatterbox Turbo TTS on %s...", device)

        local_path = snapshot_download(
            repo_id=CHATTERBOX_TURBO_HF_REPO,
            token=None,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )

        if device == "cpu":
            orig = torch.load

            def _patched(*args, **kwargs):
                kwargs.setdefault("map_location", "cpu")
                return orig(*args, **kwargs)

            with ChatterboxTurboTTSBackend._load_lock:
                torch.load = _patched
                try:
                    model = ChatterboxTurboTTS.from_local(local_path, device)
                finally:
                    torch.load = orig
        else:
            model = ChatterboxTurboTTS.from_local(local_path, device)

        patch_chatterbox_f32(model)
        self.model = model
        logger.info("Chatterbox Turbo TTS loaded")

    def unload_model(self) -> None:
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            empty_device_cache(device or "cpu")
            logger.info("Chatterbox Turbo unloaded")

    def create_voice_prompt(self, audio_path: str, reference_text: str) -> dict:
        return {"ref_audio": str(audio_path), "ref_text": reference_text or ""}

    def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
        exaggeration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        self.load_model()
        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("Reference audio not found: %s", ref_audio)
            ref_audio = None

        import torch

        if seed is not None:
            manual_seed(seed, self._device or "cpu")

        wav = self.model.generate(
            text,
            audio_prompt_path=ref_audio,
            temperature=0.8,
            top_k=1000,
            top_p=0.95,
            repetition_penalty=1.2,
        )
        if isinstance(wav, torch.Tensor):
            audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        else:
            audio = np.asarray(wav, dtype=np.float32).squeeze()
        sample_rate = getattr(self.model, "sr", None) or getattr(self.model, "sample_rate", 24000)
        return audio, int(sample_rate)
