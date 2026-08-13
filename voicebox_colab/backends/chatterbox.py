"""
Chatterbox Multilingual backend — ported from Voicebox
backend/backends/chatterbox_backend.py.
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

CHATTERBOX_HF_REPO = "ResembleAI/chatterbox"
_MTL_WEIGHT_FILES = ["t3_mtl23ls_v2.safetensors", "s3gen.pt", "ve.pt"]


class ChatterboxTTSBackend:
    _load_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self._device: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def is_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(CHATTERBOX_HF_REPO, required_files=_MTL_WEIGHT_FILES)

    def load_model(self, model_size: str = "default") -> None:
        if self.model is not None:
            return
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = get_torch_device()
        self._device = device
        logger.info("Loading Chatterbox Multilingual TTS on %s...", device)

        if device == "cpu":
            orig = torch.load

            def _patched(*args, **kwargs):
                kwargs.setdefault("map_location", "cpu")
                return orig(*args, **kwargs)

            with ChatterboxTTSBackend._load_lock:
                torch.load = _patched
                try:
                    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                finally:
                    torch.load = orig
        else:
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)

        t3_tfmr = model.t3.tfmr
        if hasattr(t3_tfmr, "config") and hasattr(t3_tfmr.config, "_attn_implementation"):
            t3_tfmr.config._attn_implementation = "eager"
            for layer in getattr(t3_tfmr, "layers", []):
                if hasattr(layer, "self_attn"):
                    layer.self_attn._attn_implementation = "eager"

        patch_chatterbox_f32(model)
        self.model = model
        logger.info("Chatterbox Multilingual TTS loaded")

    def unload_model(self) -> None:
        if self.model is not None:
            device = self._device
            del self.model
            self.model = None
            self._device = None
            empty_device_cache(device or "cpu")
            logger.info("Chatterbox unloaded")

    def create_voice_prompt(self, audio_path: str, reference_text: str) -> dict:
        return {"ref_audio": str(audio_path), "ref_text": reference_text or ""}

    # Voicebox chatterbox_backend.py defaults
    _LANG_DEFAULTS: ClassVar[dict] = {
        "he": {
            "exaggeration": 0.4,
            "cfg_weight": 0.7,
            "temperature": 0.65,
            "repetition_penalty": 2.5,
        },
    }
    _GLOBAL_DEFAULTS: ClassVar[dict] = {
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "repetition_penalty": 2.0,
    }

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

        lang_defaults = self._LANG_DEFAULTS.get(language, self._GLOBAL_DEFAULTS)
        exag = float(exaggeration) if exaggeration is not None else lang_defaults["exaggeration"]

        import torch

        if seed is not None:
            manual_seed(seed, self._device or "cpu")

        wav = self.model.generate(
            text,
            language_id=language,
            audio_prompt_path=ref_audio,
            exaggeration=exag,
            cfg_weight=lang_defaults["cfg_weight"],
            temperature=lang_defaults["temperature"],
            repetition_penalty=lang_defaults["repetition_penalty"],
        )
        if isinstance(wav, torch.Tensor):
            audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        else:
            audio = np.asarray(wav, dtype=np.float32).squeeze()
        sample_rate = getattr(self.model, "sr", None) or getattr(self.model, "sample_rate", 24000)
        return audio, int(sample_rate)
