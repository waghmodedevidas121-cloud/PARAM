"""
HumeAI TADA backend — ported from Voicebox backend/backends/hume_backend.py.

Variants:
  tada-1b     English     ~4 GB VRAM
  tada-3b-ml  10 langs    ~8 GB VRAM

Colab notes (not invented — from Voicebox source):
  * Needs the DAC Snake1d shim (utils/dac_shim.py)
  * Tokenizer is hardcoded to gated meta-llama/Llama-3.2-1B; Voicebox
    downloads unsloth/Llama-3.2-1B and patches AlignerConfig
  * 3B is tight on GPUs under 8 GB
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from .base import empty_device_cache, get_torch_device, is_model_cached, manual_seed

logger = logging.getLogger(__name__)

TADA_CODEC_REPO = "HumeAI/tada-codec"
TADA_1B_REPO = "HumeAI/tada-1b"
TADA_3B_ML_REPO = "HumeAI/tada-3b-ml"
TADA_MODEL_REPOS = {"1B": TADA_1B_REPO, "3B": TADA_3B_ML_REPO}


class HumeTadaBackend:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.model_size = "1B"
        self._device: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def is_cached(self, model_size: str = "1B") -> bool:
        repo = TADA_MODEL_REPOS.get(model_size, TADA_1B_REPO)
        return is_model_cached(repo, required_files=["model.safetensors"]) and is_model_cached(
            TADA_CODEC_REPO, required_files=["encoder/model.safetensors"]
        )

    def load_model(self, model_size: str = "1B") -> None:
        if self.model is not None and self.model_size == model_size:
            return
        if self.model is not None:
            self.unload_model()
        self.model_size = model_size

        from .dac_shim import install_dac_shim

        install_dac_shim()

        import torch
        from huggingface_hub import snapshot_download

        device = get_torch_device()
        self._device = device
        repo = TADA_MODEL_REPOS.get(model_size, TADA_1B_REPO)
        logger.info("Loading HumeAI TADA %s on %s...", model_size, device)

        snapshot_download(
            repo_id=TADA_CODEC_REPO,
            token=None,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.bin"],
        )
        snapshot_download(
            repo_id=repo,
            token=None,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.bin", "*.model"],
        )
        tokenizer_path = snapshot_download(
            repo_id="unsloth/Llama-3.2-1B",
            token=None,
            allow_patterns=["tokenizer*", "special_tokens*"],
        )

        bf16_ok = False
        if device == "cuda":
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                bf16_ok = False
        model_dtype = torch.bfloat16 if bf16_ok else torch.float32

        from tada.modules.aligner import AlignerConfig

        AlignerConfig.tokenizer_name = tokenizer_path

        from tada.modules.encoder import Encoder

        self.encoder = Encoder.from_pretrained(TADA_CODEC_REPO, subfolder="encoder").to(device)
        self.encoder.eval()

        from tada.modules.tada import TadaConfig, TadaForCausalLM

        config = TadaConfig.from_pretrained(repo)
        config.tokenizer_name = tokenizer_path
        self.model = TadaForCausalLM.from_pretrained(repo, config=config, torch_dtype=model_dtype).to(
            device
        )
        self.model.eval()
        logger.info("HumeAI TADA %s loaded", model_size)

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.encoder is not None:
            del self.encoder
            self.encoder = None
        device = self._device
        self._device = None
        if device:
            empty_device_cache(device)
        logger.info("HumeAI TADA unloaded")

    def create_voice_prompt(self, audio_path: str, reference_text: str) -> dict:
        self.load_model(self.model_size)
        import soundfile as sf
        import torch

        device = self._device
        audio_np, sr = sf.read(str(audio_path), dtype="float32")
        audio = torch.from_numpy(audio_np).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.T
        audio = audio.to(device)
        text_arg = [reference_text] if reference_text else None
        with torch.inference_mode():
            prompt = self.encoder(audio, text=text_arg, sample_rate=sr)

        prompt_dict = {}
        for field_name in prompt.__dataclass_fields__:
            val = getattr(prompt, field_name)
            if isinstance(val, torch.Tensor):
                prompt_dict[field_name] = val.detach().cpu()
            else:
                prompt_dict[field_name] = val
        return prompt_dict

    def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
        exaggeration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        self.load_model(self.model_size)
        import torch
        from tada.modules.encoder import EncoderOutput

        if seed is not None:
            manual_seed(seed, self._device or "cpu")

        device = self._device
        restored = {}
        for key, val in voice_prompt.items():
            if isinstance(val, torch.Tensor):
                if val.is_floating_point():
                    model_dtype = next(self.model.parameters()).dtype
                    restored[key] = val.to(device=device, dtype=model_dtype)
                else:
                    restored[key] = val.to(device=device)
            else:
                restored[key] = val
        prompt = EncoderOutput(**restored)
        output = self.model.generate(prompt=prompt, text=text)
        if output.audio and output.audio[0] is not None:
            audio = output.audio[0].detach().cpu().numpy().squeeze().astype(np.float32)
        else:
            logger.warning("[TADA] Generation produced no audio")
            audio = np.zeros(24000, dtype=np.float32)
        return audio, 24000
