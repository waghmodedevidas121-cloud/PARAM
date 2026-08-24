"""T4-friendly image and video upscaler backend for the PARAM Colab app.

The module deliberately keeps model imports lazy: Colab can install the
packages in one cell before this file is imported, and a notebook can be
opened without downloading several hundred megabytes of weights up front.
"""

from __future__ import annotations

import gc
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch


# Colab has /content; the fallback makes the source file harmless to import in
# a normal local checkout as well.
_default_home = Path("/content/PARAM_Upscaler") if Path("/content").exists() else Path.cwd() / "PARAM_Upscaler"
ROOT = Path(os.environ.get("PARAM_UPSCALER_HOME", str(_default_home)))
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg"}
MAX_IMAGE_PIXELS = 64_000_000

# These are the official release assets used by the Real-ESRGAN project.
MODEL_SPECS = {
    "photo": {
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "anime": {
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
    "face": {
        "filename": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.4.pth",
    },
}

ProgressFn = Optional[Callable[..., Any]]
_UPSAMPLER_CACHE: dict[tuple[str, int, str, bool], Any] = {}
_FACE_CACHE: dict[tuple[str, int, str, bool], Any] = {}


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def system_info() -> str:
    """Return a short runtime message suitable for the top of the UI."""

    device = _device()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram = props.total_memory / (1024**3)
        return f"GPU: {torch.cuda.get_device_name(device)} · VRAM: {vram:.1f} GB · FP16: enabled"
    return "GPU: not detected · running on CPU (very slow). Enable a Colab T4 runtime."


def _report(progress: ProgressFn, value: float, description: str) -> None:
    if progress is None:
        return
    value = max(0.0, min(1.0, float(value)))
    try:
        progress(value, desc=description)
    except TypeError:
        # Keeps the backend compatible with simple callback functions used in
        # tests or custom frontends.
        try:
            progress(value, description)
        except Exception:
            pass
    except Exception:
        # A progress UI must never abort a completed inference.
        pass


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return stem or "input"


def _as_path(value: Any) -> Optional[Path]:
    """Accept Gradio filepath strings, FileData dictionaries, and file objects."""

    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    elif not isinstance(value, (str, bytes, os.PathLike)):
        value = getattr(value, "path", None) or getattr(value, "name", None)
    if value is None:
        return None
    return Path(value)


def _scale_value(value: Any) -> int:
    text = str(value).lower().strip()
    if value in (2, 2.0) or text in {"2", "2x"}:
        return 2
    if value in (4, 4.0) or text in {"4", "4x"}:
        return 4
    raise ValueError("Scale must be 2x or 4x.")


def _tile_value(value: Any) -> int:
    try:
        tile = int(value)
    except (TypeError, ValueError):
        tile = 256
    if tile < 0:
        raise ValueError("Tile size cannot be negative.")
    # 0 is supported by RealESRGANer, but 128–512 is safer on a free T4.
    return tile


def _download_asset(kind: str) -> Path:
    spec = MODEL_SPECS[kind]
    destination = MODEL_DIR / spec["filename"]
    # A partial download is written to .part and therefore never mistaken for
    # a usable checkpoint after a disconnected Colab session.
    if destination.exists() and destination.stat().st_size > 10_000_000:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = Request(spec["url"], headers={"User-Agent": "PARAM-Colab-Upscaler/1.0"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if partial.stat().st_size <= 10_000_000:
            raise RuntimeError(f"Downloaded checkpoint is unexpectedly small: {partial.stat().st_size} bytes")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {spec['filename']}. Check the Colab internet connection and try again."
        ) from None
    return destination


def resolve_model(mode: Any) -> str:
    """Resolve the UI label to a checkpoint family.

    Auto intentionally defaults to the photo model. That conservative choice
    avoids silently applying an anime model to a real photo and inventing
    cartoon-like edges. Users can select Anime / illustration explicitly.
    """

    label = str(mode or "").lower()
    if "anime" in label or "illustration" in label or label == "anime":
        return "anime"
    return "photo"


def _clear_model_caches() -> None:
    _FACE_CACHE.clear()
    _UPSAMPLER_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_torchvision_compat() -> None:
    """Bridge a module path removed by newer torchvision releases."""

    try:
        import torchvision.transforms.functional_tensor  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        import torchvision.transforms.functional as functional

        sys.modules.setdefault("torchvision.transforms.functional_tensor", functional)


def _rrdb_model(model_key: str) -> Any:
    _ensure_torchvision_compat()
    from basicsr.archs.rrdbnet_arch import RRDBNet

    if model_key == "anime":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4,
        )
    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )


def get_upsampler(mode: Any = "Auto (recommended)", tile_size: Any = 256) -> tuple[Any, str]:
    """Load and cache a Real-ESRGAN upsampler, returning it and its model key."""

    _ensure_torchvision_compat()
    from realesrgan import RealESRGANer

    model_key = resolve_model(mode)
    tile = _tile_value(tile_size)
    device = _device()
    half = device.type == "cuda"
    cache_key = (model_key, tile, str(device), half)
    if cache_key in _UPSAMPLER_CACHE:
        return _UPSAMPLER_CACHE[cache_key], model_key

    # Keep only one tile/model configuration in VRAM. This matters when a user
    # changes the dropdown repeatedly in a 16 GB T4 session.
    _clear_model_caches()
    checkpoint = _download_asset(model_key)
    model = _rrdb_model(model_key)
    kwargs = dict(
        scale=4,
        model_path=str(checkpoint),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
    )
    try:
        upsampler = RealESRGANer(device=device, **kwargs)
    except TypeError:
        # Compatibility with older Real-ESRGAN wheels whose constructor did
        # not expose the device keyword.
        upsampler = RealESRGANer(**kwargs)
    _UPSAMPLER_CACHE[cache_key] = upsampler
    return upsampler, model_key


def get_face_enhancer(mode: Any = "Auto (recommended)", tile_size: Any = 256) -> Any:
    """Load GFPGAN once, using the same tiled background upsampler."""

    _ensure_torchvision_compat()
    from gfpgan import GFPGANer

    upsampler, model_key = get_upsampler(mode, tile_size)
    device = _device()
    half = device.type == "cuda"
    tile = _tile_value(tile_size)
    cache_key = (model_key, tile, str(device), half)
    if cache_key in _FACE_CACHE:
        return _FACE_CACHE[cache_key]

    checkpoint = _download_asset("face")
    kwargs = dict(
        model_path=str(checkpoint),
        upscale=4,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )
    try:
        enhancer = GFPGANer(device=device, **kwargs)
    except TypeError:
        enhancer = GFPGANer(**kwargs)
    _FACE_CACHE[cache_key] = enhancer
    return enhancer


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path.name}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Only standard 1-channel, 3-channel, or 4-channel images are supported.")
    return image


def _target_size(image: np.ndarray, scale: int) -> tuple[int, int]:
    height, width = image.shape[:2]
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _resize_to_target(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if image.shape[1] == target_width and image.shape[0] == target_height:
        return image
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)


def _enhance_once(
    image: np.ndarray,
    scale: int,
    mode: Any,
    tile_size: int,
    face_enhance: bool,
) -> tuple[np.ndarray, str]:
    upsampler, model_key = get_upsampler(mode, tile_size)
    if face_enhance:
        enhancer = get_face_enhancer(mode, tile_size)
        _, _, output = enhancer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
    else:
        output, _ = upsampler.enhance(image, outscale=float(scale))

    target_width, target_height = _target_size(image, scale)
    output = _resize_to_target(output, target_width, target_height)
    if output.dtype != np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)
    return output, model_key


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def enhance_bgr(
    image: np.ndarray,
    scale: int,
    mode: Any = "Auto (recommended)",
    tile_size: Any = 256,
    face_enhance: bool = False,
) -> tuple[np.ndarray, str, int]:
    """Upscale one BGR image, retrying with a smaller tile after a CUDA OOM."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    scale = _scale_value(scale)
    tile = _tile_value(tile_size)
    attempts = [tile]
    if tile >= 128:
        smaller = max(64, tile // 2)
        if smaller != tile:
            attempts.append(smaller)

    last_error: Optional[BaseException] = None
    for index, attempt in enumerate(attempts):
        try:
            output, model_key = _enhance_once(image, scale, mode, attempt, bool(face_enhance))
            return output, model_key, attempt
        except RuntimeError as error:
            last_error = error
            if not _is_cuda_oom(error) or index == len(attempts) - 1:
                raise
            _clear_model_caches()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError("Upscaling failed.") from last_error


def _output_path(source: Path, scale: int, extension: str) -> Path:
    filename = f"{_safe_stem(source.stem)}_upscaled_{scale}x_{uuid.uuid4().hex[:6]}{extension}"
    return OUTPUT_DIR / filename


def _status_text(
    source: Path,
    original_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    scale: int,
    model_key: str,
    tile: int,
    face_enhance: bool,
) -> str:
    before_h, before_w = original_shape[:2]
    after_h, after_w = output_shape[:2]
    device = _device()
    device_text = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    extras = " + GFPGAN face restoration" if face_enhance else ""
    return (
        f"✅ **Done:** `{source.name}` · {before_w}×{before_h} → {after_w}×{after_h} · "
        f"{scale}x · {model_key} model{extras} · tile {tile} · {device_text}"
    )


def upscale_image(
    input_file: Any,
    scale_choice: Any = "4x",
    model_mode: Any = "Auto (recommended)",
    face_enhance: bool = False,
    tile_size: Any = 256,
    progress: ProgressFn = None,
) -> tuple[str, str]:
    """Gradio callback for images."""

    source = _as_path(input_file)
    if source is None:
        raise ValueError("Upload an image first.")
    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {source.suffix or 'unknown'}")
    scale = _scale_value(scale_choice)
    tile = _tile_value(tile_size)
    _report(progress, 0.02, "Reading image")
    image = _load_bgr(source)
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large ({image.shape[1]}×{image.shape[0]}). "
            "Use an image under 64 megapixels for the free T4 runtime."
        )
    _report(progress, 0.08, "Loading AI model (first run downloads the checkpoint)")
    output, model_key, used_tile = enhance_bgr(image, scale, model_mode, tile, bool(face_enhance))
    _report(progress, 0.94, "Saving PNG")
    destination = _output_path(source, scale, ".png")
    if not cv2.imwrite(str(destination), output, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("Could not write the upscaled PNG.")
    status = _status_text(source, image.shape, output.shape, scale, model_key, used_tile, bool(face_enhance))
    _report(progress, 1.0, "Finished")
    return str(destination), status


def _mux_video_audio(raw_video: Path, source: Path, destination: Path) -> bool:
    """Encode a browser-friendly MP4 and preserve audio where FFmpeg can."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(raw_video, destination)
        return False

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_video),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(destination),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        shutil.copy2(raw_video, destination)
        return False
    return True


def upscale_video(
    input_file: Any,
    scale_choice: Any = "2x",
    model_mode: Any = "Auto (recommended)",
    face_enhance: bool = False,
    tile_size: Any = 256,
    progress: ProgressFn = None,
) -> tuple[str, str]:
    """Gradio callback for videos; frames are processed sequentially to cap VRAM."""

    source = _as_path(input_file)
    if source is None:
        raise ValueError("Upload a video first.")
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {source.suffix or 'unknown'}")
    scale = _scale_value(scale_choice)
    tile = _tile_value(tile_size)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Could not open this video. Try MP4 or re-encode the source first.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("The video has no readable frames.")
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    target_width, target_height = width * scale, height * scale
    destination = _output_path(source, scale, ".mp4")
    work_dir = Path(tempfile.mkdtemp(prefix="param_upscale_", dir=str(ROOT)))
    raw_video = work_dir / "silent_upscaled.mp4"
    writer = None
    frame_index = 0
    model_key = resolve_model(model_mode)
    used_tile = tile

    try:
        writer = cv2.VideoWriter(
            str(raw_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (target_width, target_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the temporary output video.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index == 0:
                _report(progress, 0.02, "Loading AI model (first run downloads the checkpoint)")
            output, model_key, used_tile = enhance_bgr(
                frame,
                scale,
                model_mode,
                tile,
                bool(face_enhance),
            )
            output = _resize_to_target(output, target_width, target_height)
            writer.write(output)
            frame_index += 1
            if total > 0:
                fraction = 0.05 + 0.88 * (frame_index / total)
                _report(progress, fraction, f"Upscaling frame {frame_index}/{total}")
            elif frame_index % 5 == 0:
                _report(progress, 0.5, f"Upscaling frame {frame_index}")

        if writer is not None:
            writer.release()
            writer = None
        cap.release()
        if frame_index == 0:
            raise ValueError("No frames could be read from this video.")

        _report(progress, 0.95, "Encoding MP4 and preserving audio")
        audio_preserved = _mux_video_audio(raw_video, source, destination)
        extras = " + GFPGAN face restoration" if face_enhance else ""
        audio_note = "audio preserved" if audio_preserved else "audio unavailable"
        device = _device()
        device_text = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        status = (
            f"✅ **Done:** `{source.name}` · {width}×{height} → {target_width}×{target_height} · "
            f"{scale}x · {frame_index} frames · {model_key} model{extras} · {audio_note} · {device_text}"
        )
        _report(progress, 1.0, "Finished")
        return str(destination), status
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        shutil.rmtree(work_dir, ignore_errors=True)


def build_app() -> Any:
    """Build the Gradio interface without launching it."""

    import gradio as gr

    css = """
    .gradio-container { max-width: 1180px !important; }
    .param-title { text-align: center; margin-bottom: 0.2rem; }
    .param-subtitle { text-align: center; color: #64748b; }
    """
    with gr.Blocks(css=css, title="PARAM AI Upscaler") as demo:
        gr.Markdown("# PARAM AI Upscaler", elem_classes=["param-title"])
        gr.Markdown(
            "T4-friendly image and video enhancement · Real-ESRGAN + optional GFPGAN",
            elem_classes=["param-subtitle"],
        )
        gr.Markdown(
            f"**Runtime:** {system_info()}\n\n"
            "Auto mode uses the safer photo model. Choose Anime / illustration for cartoons or line art. "
            "Large videos can take time on a free Colab session."
        )

        with gr.Row():
            with gr.Column(scale=1):
                scale = gr.Radio(["2x", "4x"], value="4x", label="Output scale")
                model = gr.Dropdown(
                    ["Auto (recommended)", "Photo / real-world", "Anime / illustration"],
                    value="Auto (recommended)",
                    label="Content / quality mode",
                )
                face = gr.Checkbox(
                    value=False,
                    label="Restore faces with GFPGAN (slower)",
                    info="Useful for portraits; it may change identity or add detail that was not present.",
                )
                tile = gr.Slider(
                    minimum=128,
                    maximum=512,
                    value=256,
                    step=64,
                    label="Tile size (T4 VRAM safety)",
                    info="256 is the recommended free-T4 setting. Lower it if CUDA runs out of memory.",
                )

        with gr.Tab("Image"):
            with gr.Row():
                image_input = gr.File(
                    label="Upload image",
                    file_types=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
                    type="filepath",
                )
                image_output = gr.Image(label="Upscaled PNG", type="filepath")
            image_button = gr.Button("Upscale image", variant="primary")
            image_status = gr.Markdown()

        with gr.Tab("Video"):
            video_input = gr.Video(
                label="Upload video",
                sources=["upload"],
                type="filepath",
            )
            video_button = gr.Button("Upscale video", variant="primary")
            video_output = gr.Video(label="Upscaled MP4")
            video_status = gr.Markdown()
            gr.Markdown(
                "Video frames are processed sequentially to stay inside T4 VRAM. "
                "Keep the Colab tab open until the output appears."
            )

        # Defining the progress default inside build_app keeps gradio optional
        # for users who only want to import the backend in a script.
        def image_job(input_file, scale_choice, model_mode, face_enhance, tile_size, progress=gr.Progress()):
            return upscale_image(input_file, scale_choice, model_mode, face_enhance, tile_size, progress)

        def video_job(input_file, scale_choice, model_mode, face_enhance, tile_size, progress=gr.Progress()):
            return upscale_video(input_file, scale_choice, model_mode, face_enhance, tile_size, progress)

        common_inputs = [scale, model, face, tile]
        image_button.click(
            image_job,
            inputs=[image_input, *common_inputs],
            outputs=[image_output, image_status],
        )
        video_button.click(
            video_job,
            inputs=[video_input, *common_inputs],
            outputs=[video_output, video_status],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(share=True, server_name="0.0.0.0", server_port=7860)
