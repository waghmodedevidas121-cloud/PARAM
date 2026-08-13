"""
Qwen CustomVoice backend — ported from Voicebox
backend/backends/qwen_custom_voice_backend.py.

Preset speakers + generate_custom_voice() + instruct.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..capabilities import LANGUAGE_CODE_TO_NAME
from .base import empty_device_cache, get_torch_device, is_model_cached, manual_seed

logger = logging.getLogger(__name__)

# (speaker_id, display_name, gender, native_language_code, description)
QWEN_CUSTOM_VOICES = [
    ("Vivian", "Vivian", "female", "zh", "Bright, slightly edgy young female voice"),
    ("Serena", "Serena", "female", "zh", "Warm, gentle young female voice"),
    ("Uncle_Fu", "Uncle Fu", "male", "zh", "Seasoned male voice with a low, mellow timbre"),
    ("Dylan", "Dylan", "male", "zh", "Youthful Beijing male voice with a clear, natural timbre"),
    ("Eric", "Eric", "male", "zh", "Lively Chengdu male voice with a slightly husky brightness"),
    ("Ryan", "Ryan", "male", "en", "Dynamic male voice with strong rhythmic drive"),
    ("Aiden", "Aiden", "male", "en", "Sunny American male voice with a clear midrange"),
    ("Ono_Anna", "Ono Anna", "female", "ja", "Playful Japanese female voice with a light, nimble timbre"),
    ("Sohee", "Sohee", "female", "ko", "Warm Korean female voice with rich emotion"),
]

QWEN_CV_DEFAULT_SPEAKER = "Ryan"

QWEN_CV_HF_REPOS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}


def voice_choices() -> list[tuple[str, str]]:
    return [
        (f"{display}  ·  {gender}  ·  {lang}  —  {desc}", voice_id)
        for voice_id, display, gender, lang, desc in QWEN_CUSTOM_VOICES
    ]


class QwenCustomVoiceBackend:
    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self.device = get_torch_device()
        self._current_model_size: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in QWEN_CV_HF_REPOS:
            raise ValueError(f"Unknown CustomVoice size: {model_size}")
        return QWEN_CV_HF_REPOS[model_size]

    def is_cached(self, model_size: Optional[str] = None) -> bool:
        return is_model_cached(self._get_model_path(model_size or self.model_size))

    def load_model(self, model_size: Optional[str] = None) -> None:
        model_size = model_size or self.model_size
        if self.model is not None and self._current_model_size == model_size:
            return
        if self.model is not None:
            self.unload_model()

        import torch
        from qwen_tts import Qwen3TTSModel

        model_path = self._get_model_path(model_size)
        logger.info("Loading Qwen CustomVoice %s on %s...", model_size, self.device)
        if self.device == "cpu":
            self.model = Qwen3TTSModel.from_pretrained(
                model_path, torch_dtype=torch.float32, low_cpu_mem_usage=False
            )
        else:
            self.model = Qwen3TTSModel.from_pretrained(
                model_path, device_map=self.device, torch_dtype=torch.bfloat16
            )
        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen CustomVoice %s loaded", model_size)

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            empty_device_cache(self.device)
            logger.info("Qwen CustomVoice unloaded")

    def create_voice_prompt(self, audio_path: str, reference_text: str) -> dict:
        return {
            "voice_type": "preset",
            "preset_engine": "qwen_custom_voice",
            "preset_voice_id": QWEN_CV_DEFAULT_SPEAKER,
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
        speaker = voice_prompt.get("preset_voice_id") or QWEN_CV_DEFAULT_SPEAKER
        if seed is not None:
            manual_seed(seed, self.device)

        lang_name = LANGUAGE_CODE_TO_NAME.get(language, "auto")
        kwargs = {
            "text": text,
            "language": lang_name.capitalize() if lang_name != "auto" else "Auto",
            "speaker": speaker,
        }
        if instruct:
            kwargs["instruct"] = instruct
        wavs, sample_rate = self.model.generate_custom_voice(**kwargs)
        audio = wavs[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32).squeeze(), int(sample_rate)
