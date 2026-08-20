#!/usr/bin/env python3
"""Apply idempotent T4/standard-RAM compatibility patches to OmniAvatar 1.3B."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str, marker: str, label: str) -> None:
    source = path.read_text(encoding="utf-8")
    if marker in source:
        return
    if old in source:
        path.write_text(source.replace(old, new, 1), encoding="utf-8")
    else:
        raise RuntimeError(f"Could not apply {label} patch to {path}")


def apply(root: Path) -> None:
    inference = root / "scripts" / "inference.py"
    io_utils = root / "OmniAvatar" / "utils" / "io_utils.py"
    if not inference.is_file() or not io_utils.is_file():
        raise FileNotFoundError(f"OmniAvatar source was not found under {root}")

    replace_once(
        inference,
        'tea_cache_model_id="Wan2.1-T2V-14B")',
        'tea_cache_model_id="Wan2.1-T2V-1.3B" if "1.3B" in args.dit_path else "Wan2.1-T2V-14B")',
        'tea_cache_model_id="Wan2.1-T2V-1.3B" if',
        "TeaCache 1.3B",
    )

    audio_old = (
        "            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])\n"
        "        else:\n"
        "            audio_embeddings = None"
    )
    audio_new = (
        "            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])\n"
        "            del hidden_states, input_values\n"
        "            self.audio_encoder.to(\"cpu\")\n"
        "            torch.cuda.empty_cache()\n"
        "        else:\n"
        "            audio_embeddings = None"
    )
    replace_once(inference, audio_old, audio_new, 'self.audio_encoder.to("cpu")', "Wav2Vec offload")

    mmap_old = '    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)'
    mmap_new = '''    try:
        state_dict = torch.load(
            file_path, map_location="cpu", weights_only=True, mmap=True
        )
    except (RuntimeError, ValueError):
        # Compatibility fallback for legacy non-zip checkpoints.
        state_dict = torch.load(file_path, map_location="cpu", weights_only=True)'''
    replace_once(io_utils, mmap_old, mmap_new, "weights_only=True, mmap=True", "mmap checkpoint")

    staged_old = '''        # Load models
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models(
            [
                args.dit_path.split(","),
                args.text_encoder_path,
                args.vae_path
            ],
            torch_dtype=self.dtype, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
            device='cpu',
        )'''
    staged_new = '''        # T4 staged model loading: keep the large T5 memory-mapped in BF16
        # storage while all actual CUDA computation remains FP16.
        import gc
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_model(
            args.dit_path.split(","), torch_dtype=self.dtype, device="cpu"
        )
        dit_for_staging = model_manager.fetch_model("wan_video_dit")
        dit_for_staging.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        model_manager.load_model(
            args.text_encoder_path, torch_dtype=torch.bfloat16, device="cpu"
        )
        model_manager.load_model(
            args.vae_path, torch_dtype=self.dtype, device="cpu"
        )'''
    replace_once(inference, staged_old, staged_new, "# T4 staged model loading:", "staged loader")

    config = '''# T4 / Turing + standard host-RAM profile

dtype: "fp16"
text_encoder_path: pretrained_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
image_encoder_path: None
dit_path: pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
vae_path: pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
wav2vec_path: pretrained_models/wav2vec2-base-960h
exp_path: pretrained_models/OmniAvatar-1.3B
num_persistent_param_in_dit: 7000000000
reload_cfg: true
sp_size: 1
seed: 42
image_sizes_720: [[400, 720], [720, 720], [720, 400]]
image_sizes_1280: [[720, 720], [528, 960], [960, 528], [720, 1280], [1280, 720]]
max_hw: 720
max_tokens: 12000
seq_len: 200
overlap_frame: 13
guidance_scale: 4.5
audio_scale: 5.0
num_steps: 20
fps: 25
sample_rate: 16000
negative_prompt: "Vivid color tones, background or camera moving quickly, screen switching, subtitles, special effects, mutation, overexposed, static, blurred details, painting, still image, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, motionless image, chaotic or crowded background, extra limbs, walking backward"
silence_duration_s: 0.3
use_fsdp: false
tea_cache_l1_thresh: 0.0
'''
    config_path = root / "configs" / "inference_t4.yaml"
    config_path.write_text(config, encoding="utf-8")
    print(f"Applied OmniAvatar T4 standard-RAM patches under {root}")
    print(f"Wrote {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="/content/OmniAvatar")
    args = parser.parse_args()
    apply(Path(args.root).resolve())


if __name__ == "__main__":
    main()
