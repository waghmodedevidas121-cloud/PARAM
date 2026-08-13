"""
Kokoro 82M backend — ported from Voicebox backend/backends/kokoro_backend.py.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .base import empty_device_cache, get_torch_device, is_model_cached

logger = logging.getLogger(__name__)

KOKORO_HF_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
KOKORO_DEFAULT_VOICE = "af_heart"

# (voice_id, display_name, gender, lang_code) — Voicebox KOKORO_VOICES
KOKORO_VOICES = [
    ("af_alloy", "Alloy", "female", "en"),
    ("af_aoede", "Aoede", "female", "en"),
    ("af_bella", "Bella", "female", "en"),
    ("af_heart", "Heart", "female", "en"),
    ("af_jessica", "Jessica", "female", "en"),
    ("af_kore", "Kore", "female", "en"),
    ("af_nicole", "Nicole", "female", "en"),
    ("af_nova", "Nova", "female", "en"),
    ("af_river", "River", "female", "en"),
    ("af_sarah", "Sarah", "female", "en"),
    ("af_sky", "Sky", "female", "en"),
    ("am_adam", "Adam", "male", "en"),
    ("am_echo", "Echo", "male", "en"),
    ("am_eric", "Eric", "male", "en"),
    ("am_fenrir", "Fenrir", "male", "en"),
    ("am_liam", "Liam", "male", "en"),
    ("am_michael", "Michael", "male", "en"),
    ("am_onyx", "Onyx", "male", "en"),
    ("am_puck", "Puck", "male", "en"),
    ("am_santa", "Santa", "male", "en"),
    ("bf_alice", "Alice", "female", "en"),
    ("bf_emma", "Emma", "female", "en"),
    ("bf_isabella", "Isabella", "female", "en"),
    ("bf_lily", "Lily", "female", "en"),
    ("bm_daniel", "Daniel", "male", "en"),
    ("bm_fable", "Fable", "male", "en"),
    ("bm_george", "George", "male", "en"),
    ("bm_lewis", "Lewis", "male", "en"),
    ("ef_dora", "Dora", "female", "es"),
    ("em_alex", "Alex", "male", "es"),
    ("em_santa", "Santa", "male", "es"),
    ("ff_siwis", "Siwis", "female", "fr"),
    ("hf_alpha", "Alpha", "female", "hi"),
    ("hf_beta", "Beta", "female", "hi"),
    ("hm_omega", "Omega", "male", "hi"),
    ("hm_psi", "Psi", "male", "hi"),
    ("if_sara", "Sara", "female", "it"),
    ("im_nicola", "Nicola", "male", "it"),
    ("jf_alpha", "Alpha", "female", "ja"),
    ("jf_gongitsune", "Gongitsune", "female", "ja"),
    ("jf_nezumi", "Nezumi", "female", "ja"),
    ("jf_tebukuro", "Tebukuro", "female", "ja"),
    ("jm_kumo", "Kumo", "male", "ja"),
    ("pf_dora", "Dora", "female", "pt"),
    ("pm_alex", "Alex", "male", "pt"),
    ("pm_santa", "Santa", "male", "pt"),
    ("zf_xiaobei", "Xiaobei", "female", "zh"),
    ("zf_xiaoni", "Xiaoni", "female", "zh"),
    ("zf_xiaoxiao", "Xiaoxiao", "female", "zh"),
    ("zf_xiaoyi", "Xiaoyi", "female", "zh"),
    ("zm_yunjian", "Yunjian", "male", "zh"),
    ("zm_yunxi", "Yunxi", "male", "zh"),
    ("zm_yunxia", "Yunxia", "male", "zh"),
    ("zm_yunyang", "Yunyang", "male", "zh"),
]

LANG_CODE_MAP = {
    "en": "a",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "ja": "j",
    "zh": "z",
}


def voice_choices(language: Optional[str] = None) -> list[tuple[str, str]]:
    out = []
    for voice_id, display, gender, lang in KOKORO_VOICES:
        if language and lang != language:
            continue
        out.append((f"{display}  ·  {gender}  ·  {lang}", voice_id))
    return out or [(f"{KOKORO_DEFAULT_VOICE}", KOKORO_DEFAULT_VOICE)]


class KokoroTTSBackend:
    def __init__(self):
        self._model = None
        self._pipelines: dict = {}
        self._device: Optional[str] = None
        self.model_size = "default"

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = get_torch_device()
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None

    def is_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(KOKORO_HF_REPO, required_files=["config.json", "kokoro-v1_0.pth"])

    def load_model(self, model_size: str = "default") -> None:
        if self._model is not None:
            return
        from kokoro import KModel

        device = self.device
        logger.info("Loading Kokoro-82M on %s...", device)
        self._model = KModel(repo_id=KOKORO_HF_REPO).to(device).eval()
        logger.info("Kokoro-82M loaded")

    def _get_pipeline(self, lang_code: str):
        kokoro_lang = LANG_CODE_MAP.get(lang_code, "a")
        if kokoro_lang not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[kokoro_lang] = KPipeline(
                lang_code=kokoro_lang,
                repo_id=KOKORO_HF_REPO,
                model=self._model,
            )
        return self._pipelines[kokoro_lang]

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._pipelines.clear()
            empty_device_cache(self.device)
            logger.info("Kokoro unloaded")

    def create_voice_prompt(self, audio_path: str, reference_text: str) -> dict:
        return {
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": KOKORO_DEFAULT_VOICE,
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
        voice_name = (
            voice_prompt.get("preset_voice_id")
            or voice_prompt.get("kokoro_voice")
            or KOKORO_DEFAULT_VOICE
        )
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        pipeline = self._get_pipeline(language)
        audio_chunks = []
        for result in pipeline(text, voice=voice_name, speed=1.0):
            if result.audio is not None:
                chunk = result.audio
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                audio_chunks.append(np.asarray(chunk).squeeze())
        if not audio_chunks:
            return np.zeros(KOKORO_SAMPLE_RATE, dtype=np.float32), KOKORO_SAMPLE_RATE
        return np.concatenate(audio_chunks).astype(np.float32), KOKORO_SAMPLE_RATE
