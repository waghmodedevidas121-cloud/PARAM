"""
Colab system probe + VRAM recommendations.

Thresholds come from Voicebox docs/content/docs/developer/model-management.mdx,
not guesses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .capabilities import MODEL_VRAM_GB


@dataclass
class SystemInfo:
    gpu_name: str
    cuda_available: bool
    vram_total_gb: float
    vram_free_gb: float
    cuda_version: str
    device: str
    recommendation: str
    recommended_models: list[str]
    notes: list[str]


def probe_system() -> SystemInfo:
    gpu_name = "None"
    cuda_available = False
    vram_total = 0.0
    vram_free = 0.0
    cuda_version = "n/a"
    device = "cpu"
    notes: list[str] = []

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device = "cuda"
            props = torch.cuda.get_device_properties(0)
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = props.total_memory / (1024**3)
            try:
                free, _total = torch.cuda.mem_get_info(0)
                vram_free = free / (1024**3)
            except Exception:
                vram_free = vram_total
            if torch.version.cuda:
                cuda_version = str(torch.version.cuda)
        else:
            notes.append("CUDA is not available. Generation will fall back to CPU (slow).")
    except ImportError:
        notes.append("PyTorch is not installed.")

    recommended = recommend_models(vram_total if cuda_available else 0.0)
    recommendation = _recommendation_blurb(vram_total if cuda_available else 0.0, gpu_name)
    return SystemInfo(
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        vram_total_gb=round(vram_total, 2),
        vram_free_gb=round(vram_free, 2),
        cuda_version=cuda_version,
        device=device,
        recommendation=recommendation,
        recommended_models=recommended,
        notes=notes,
    )


def recommend_models(vram_gb: float) -> list[str]:
    """Return model_names whose official VRAM requirement fits with ~0.5 GB headroom."""
    if vram_gb <= 0:
        return ["kokoro", "luxtts"]  # Voicebox documents both as CPU-friendly
    fitted = []
    for name, need in sorted(MODEL_VRAM_GB.items(), key=lambda kv: kv[1]):
        if need + 0.5 <= vram_gb:
            fitted.append(name)
    return fitted or ["kokoro"]


def _recommendation_blurb(vram_gb: float, gpu_name: str) -> str:
    # Official table:
    #   Kokoro ~150 MB, LuxTTS ~1 GB, Turbo ~1.5 GB, Qwen 0.6B ~2 GB,
    #   Chatterbox MTL ~3 GB, TADA 1B ~4 GB, Qwen 1.7B ~6 GB, TADA 3B ~8 GB
    if vram_gb <= 0:
        return "No GPU — use Kokoro or LuxTTS (Voicebox documents both as CPU-friendly)."
    if vram_gb < 2.0:
        return f"{gpu_name} · {vram_gb:.1f} GB — lightweight only (Kokoro, LuxTTS)."
    if vram_gb < 4.0:
        return (
            f"{gpu_name} · {vram_gb:.1f} GB — Qwen 0.6B / CustomVoice 0.6B, "
            "Chatterbox Turbo, Chatterbox Multilingual, Kokoro, LuxTTS."
        )
    if vram_gb < 6.0:
        return (
            f"{gpu_name} · {vram_gb:.1f} GB — everything below 6 GB, including TADA 1B. "
            "Unload before switching."
        )
    if vram_gb < 8.0:
        return (
            f"{gpu_name} · {vram_gb:.1f} GB — Qwen 1.7B / CustomVoice 1.7B fit. "
            "TADA 3B (~8 GB) is tight."
        )
    return (
        f"{gpu_name} · {vram_gb:.1f} GB — any single Voicebox engine, including TADA 3B. "
        "Still load only one model at a time."
    )


def system_as_dict() -> dict[str, Any]:
    return asdict(probe_system())


def format_status_markdown(current_model: str = "—", status: str = "Ready") -> str:
    info = probe_system()
    cuda = "Available" if info.cuda_available else "Not available"
    notes = ""
    if info.notes:
        notes = "\n\n" + "\n".join(f"- {n}" for n in info.notes)
    return f"""### COLAB SYSTEM

| | |
|---|---|
| **GPU** | {info.gpu_name} |
| **CUDA** | {cuda} ({info.cuda_version}) |
| **VRAM** | {info.vram_total_gb:.1f} GB total · {info.vram_free_gb:.1f} GB free |
| **Device** | `{info.device}` |
| **Voicebox Engine** | {status} |
| **Current Model** | {current_model} |
| **Status** | ● {status} |

{info.recommendation}{notes}
"""


def empty_cuda_cache() -> str:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            free, total = torch.cuda.mem_get_info(0)
            return (
                f"GPU cache cleared. "
                f"{free / 1024**3:.2f} / {total / 1024**3:.2f} GB free."
            )
        return "No CUDA device — nothing to clear."
    except Exception as exc:
        return f"Failed to clear GPU memory: {exc}"
