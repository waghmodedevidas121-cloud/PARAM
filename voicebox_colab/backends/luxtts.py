"""
LuxTTS backend — ported from Voicebox backend/backends/luxtts_backend.py.

English-only zero-shot cloning. ~1 GB VRAM, 48 kHz.
Git deps (Zipvoice / linacodec) are optional on Colab — the engine is
hidden if the import fails.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .base import empty_device_cache, get_torch_device, is_model_cached, manual_seed

logger = logging.getLogger(__name__)

LUXTTS_HF_REPO = "YatharthS/LuxTTS"


class LuxTTSBackend:
    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device: Optional[str] = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = get_torch_device()
        return self._device

    def is_loaded(self) -> bool:
        return self.model is not None

    def is_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(
            LUXTTS_HF_REPO, weight_extensions=(".pt", ".safetensors", ".onnx", ".bin")
        )

    def load_model(self, model_size: str = "default") -> None:
        if self.model is not None:
            return
        from zipvoice.luxvoice import LuxTTS

        device = self.device
        logger.info("Loading LuxTTS on %s...", device)
        if device == "cpu":
            import os

            threads = os.cpu_count() or 4
            self.model = LuxTTS(
                model_path=LUXTTS_HF_REPO, device="cpu", threads=min(threads, 8)
            )
        else:
            self.model = LuxTTS(model_path=LUXTTS_HF_REPO, device=device)
        logger.info("LuxTTS loaded")

    def unload_model(self) -> None:
        if self.model is not None:
            device = self.device
            del self.model
            self.model = None
            self._device = None
            empty_device_cache(device)
            logger.info("LuxTTS unloaded")

    def create_voice_prompt(self, audio_path: str, reference_text: str) -> dict:
        self.load_model()
        return self.model.encode_prompt(prompt_audio=str(audio_path), duration=5, rms=0.01)

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
        wav = self.model.generate_speech(
            text=text,
            encode_dict=voice_prompt,
            num_steps=4,
            guidance_scale=3.0,
            t_shift=0.5,
            speed=1.0,
            return_smooth=False,
        )
        audio = wav.detach().cpu().numpy().squeeze()
        return np.asarray(audio, dtype=np.float32), 48000
