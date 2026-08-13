"""
Qwen3-TTS Base backend — ported from Voicebox backend/backends/pytorch_backend.py.

Uses Qwen/Qwen3-TTS-12Hz-{1.7B,0.6B}-Base and generate_voice_clone().
supports_instruct is False on the Base checkpoint (drops instruct silently).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..capabilities import LANGUAGE_CODE_TO_NAME
from .base import empty_device_cache, get_torch_device, is_model_cached, manual_seed

logger = logging.getLogger(__name__)

HF_REPOS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}


class QwenTTSBackend:
    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self.device = get_torch_device()
        self._current_model_size: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in HF_REPOS:
            raise ValueError(f"Unknown Qwen model size: {model_size}")
        return HF_REPOS[model_size]

    def is_cached(self, model_size: Optional[str] = None) -> bool:
        return is_model_cached(self._get_model_path(model_size or self.model_size))

    def load_model(self, model_size: Optional[str] = None) -> None:
        model_size = model_size or self.model_size
        if self.model is not None and self._current_model_size == model_size:
            return
        if self.model is not None:
            self.unload_model()

        import torch
        from huggingface_hub import constants as hf_constants
        from qwen_tts import Qwen3TTSModel

        model_path = self._get_model_path(model_size)
        logger.info("Loading Qwen TTS %s on %s...", model_size, self.device)
        cache_dir = hf_constants.HF_HUB_CACHE
        if self.device == "cpu":
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
        else:
            self.model = Qwen3TTSModel.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen TTS %s loaded", model_size)

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            empty_device_cache(self.device)
            logger.info("Qwen TTS unloaded")

    def create_voice_prompt(self, audio_path: str, reference_text: str) -> dict:
        self.load_model()
        return self.model.create_voice_clone_prompt(
            ref_audio=str(audio_path),
            ref_text=reference_text or "",
            x_vector_only_mode=False,
        )

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
        if seed is not None:
            manual_seed(seed, self.device)
        # instruct is accepted for protocol compatibility but the Base
        # checkpoint drops it (ModelConfig.supports_instruct=False).
        wavs, sample_rate = self.model.generate_voice_clone(
            text=text,
            voice_clone_prompt=voice_prompt,
            language=LANGUAGE_CODE_TO_NAME.get(language, "auto"),
            instruct=None,
        )
        audio = wavs[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32).squeeze(), int(sample_rate)
