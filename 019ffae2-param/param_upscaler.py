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
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen

import ctypes
import cv2
import numpy as np
import torch

# T4-friendly inference defaults. cuDNN benchmarking avoids repeatedly
# selecting kernels for the same tile shape across video frames.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


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
    "photo_x2": {
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
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

MODEL_LABELS = {
    "photo": "RealESRGAN_x4plus",
    "photo_x2": "RealESRGAN_x2plus",
    "anime": "RealESRGAN_x4plus_anime_6B",
}

VIDEO2X_VERSION = "6.4.0"
VIDEO2X_APPIMAGE_URL = (
    "https://github.com/k4yt3x/video2x/releases/download/"
    f"{VIDEO2X_VERSION}/Video2X-x86_64.AppImage"
)
VIDEO2X_DIR = ROOT / "video2x"
VIDEO2X_APPIMAGE = VIDEO2X_DIR / f"Video2X-{VIDEO2X_VERSION}.AppImage"
_VIDEO2X_RUNTIME: Optional[tuple[list[str], Path, dict[str, str]]] = None
_VIDEO2X_DEVICE_INDEX = 0


def _model_label(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key)


ProgressFn = Optional[Callable[..., Any]]
_UPSAMPLER_CACHE: dict[tuple[str, int, str, bool], Any] = {}
_FACE_CACHE: dict[tuple[str, int, str, bool], Any] = {}


def _device() -> torch.device:
    if torch.cuda.is_available():
        # Colab's first visible GPU is the T4 target. Selecting it explicitly
        # prevents a stale/default device setting from silently falling back.
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def _require_cuda() -> torch.device:
    device = _device()
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA GPU is not available. In Colab choose Runtime → Change runtime type → T4 GPU, "
            "then restart the runtime and run the notebook again."
        )
    return device


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


def _ensure_vulkan_runtime(progress: ProgressFn = None) -> None:
    """Install the Vulkan loader that is absent from some Colab images."""

    try:
        ctypes.CDLL("libvulkan.so.1")
        return
    except OSError:
        pass

    apt_get = shutil.which("apt-get")
    if apt_get is None:
        raise RuntimeError("libvulkan.so.1 is missing and apt-get is unavailable in this runtime.")
    _report(progress, 0.01, "Installing Vulkan runtime for Video2X")
    update = subprocess.run(
        [apt_get, "update", "-qq"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    install = subprocess.run(
        [apt_get, "install", "-y", "-qq", "libvulkan1", "vulkan-tools", "mesa-vulkan-drivers"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if update.returncode != 0 or install.returncode != 0:
        details = "\n".join((install.stdout or update.stdout or "").splitlines()[-8:])
        raise RuntimeError("Could not install the Vulkan runtime in Colab.\n" + details)
    try:
        ctypes.CDLL("libvulkan.so.1")
    except OSError as error:
        raise RuntimeError("Vulkan loader installation finished, but libvulkan.so.1 is still unavailable.") from error


def _video2x_vulkan_device(command: list[str], cwd: Path, env: dict[str, str]) -> Optional[int]:
    """List Video2X Vulkan devices and prefer NVIDIA over a software ICD."""

    result = subprocess.run(
        command + ["--list-devices"],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
        check=False,
    )
    output = result.stdout or ""
    if output.strip():
        print("[Video2X] Vulkan devices:\n" + output.strip(), flush=True)
    devices = []
    for match in re.finditer(r"(?m)^\s*(\d+)[.)]\s*(.+)$", output):
        devices.append((int(match.group(1)), match.group(2).strip().lower()))
    if not devices:
        # Older Video2X builds may not prefix the first device with an index.
        if "nvidia" in output.lower() or "tesla" in output.lower() or "t4" in output.lower():
            return 0
        return None
    for index, label in devices:
        if any(token in label for token in ("nvidia", "tesla", "t4", "rtx", "quadro")):
            return index
    return devices[0][0]


def _download_video2x_appimage(progress: ProgressFn = None) -> Path:
    """Download the official Linux Video2X bundle once per Colab runtime."""

    VIDEO2X_DIR.mkdir(parents=True, exist_ok=True)
    if VIDEO2X_APPIMAGE.exists() and VIDEO2X_APPIMAGE.stat().st_size > 150_000_000:
        return VIDEO2X_APPIMAGE

    partial = VIDEO2X_APPIMAGE.with_suffix(VIDEO2X_APPIMAGE.suffix + ".part")
    partial.unlink(missing_ok=True)
    _report(progress, 0.02, "Downloading Video2X 6.4.0 (first video run only)")
    request = Request(VIDEO2X_APPIMAGE_URL, headers={"User-Agent": "PARAM-Colab-Upscaler/1.0"})
    try:
        with urlopen(request, timeout=180) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=4 * 1024 * 1024)
        if partial.stat().st_size <= 150_000_000:
            raise RuntimeError("Video2X AppImage download is unexpectedly small")
        partial.replace(VIDEO2X_APPIMAGE)
        VIDEO2X_APPIMAGE.chmod(VIDEO2X_APPIMAGE.stat().st_mode | 0o111)
    except Exception:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the Video2X Linux AppImage. Check the Colab internet connection and retry."
        ) from None
    return VIDEO2X_APPIMAGE


def _probe_video2x(command: list[str], cwd: Path, env: dict[str, str]) -> bool:
    try:
        result = subprocess.run(
            command + ["--version"],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "video2x" in result.stdout.lower()


def prepare_video2x(progress: ProgressFn = None) -> str:
    """Prepare the official Video2X AppImage and return a runnable command.

    Video2X 6 is a C++/Vulkan application. The AppImage contains its ncnn
    models and libraries, while Colab supplies the NVIDIA Vulkan driver.
    """

    global _VIDEO2X_RUNTIME, _VIDEO2X_DEVICE_INDEX
    _ensure_vulkan_runtime(progress)
    if _VIDEO2X_RUNTIME is not None:
        return " ".join(_VIDEO2X_RUNTIME[0])

    appimage = _download_video2x_appimage(progress)
    appimage.chmod(appimage.stat().st_mode | 0o111)
    env = os.environ.copy()
    direct_command = [str(appimage), "--appimage-extract-and-run"]
    if _probe_video2x(direct_command, VIDEO2X_DIR, env):
        device_index = _video2x_vulkan_device(direct_command, VIDEO2X_DIR, env)
        if device_index is not None:
            _VIDEO2X_DEVICE_INDEX = device_index
            _VIDEO2X_RUNTIME = (direct_command, VIDEO2X_DIR, env)
            _report(progress, 0.12, f"Video2X Vulkan backend ready (device {device_index})")
            return " ".join(direct_command)

    # FUSE is not available in some Colab runtimes. Extracting the AppImage
    # uses the same bundled binary without requiring a FUSE mount.
    extracted = VIDEO2X_DIR / "squashfs-root"
    if not extracted.exists():
        result = subprocess.run(
            [str(appimage), "--appimage-extract"],
            cwd=str(VIDEO2X_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0 or not extracted.exists():
            raise RuntimeError(
                "Video2X AppImage could not start in this Colab runtime. "
                "Try Runtime → Factory reset runtime and run the setup cells again."
            )
    runner = extracted / "AppRun"
    if not runner.exists():
        candidates = [p for p in extracted.rglob("video2x") if p.is_file() and os.access(p, os.X_OK)]
        if not candidates:
            raise RuntimeError("Video2X executable was not found inside the AppImage.")
        runner = candidates[0]
    env["APPDIR"] = str(extracted)
    command = [str(runner)]
    if not _probe_video2x(command, extracted, env):
        raise RuntimeError("Video2X started, but its Vulkan CLI probe failed.")
    device_index = _video2x_vulkan_device(command, extracted, env)
    if device_index is None:
        raise RuntimeError(
            "Video2X could not find a Vulkan GPU. Install the Vulkan runtime and confirm that the T4 is visible."
        )
    _VIDEO2X_DEVICE_INDEX = device_index
    _VIDEO2X_RUNTIME = (command, extracted, env)
    _report(progress, 0.12, f"Video2X Vulkan backend ready (device {device_index})")
    return " ".join(command)


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


def _ensure_source_package_compat() -> None:
    """Make source-only Real-ESRGAN/BasicSR clones importable.

    Their ``version.py`` files are generated by legacy ``setup.py`` commands.
    The Colab installer intentionally skips those commands, so create the tiny
    metadata files and avoid eagerly importing unused training modules.
    """

    source_specs = (
        (
            Path("/content/Real-ESRGAN"),
            "realesrgan",
            "0.3.0",
            "from .utils import RealESRGANer\n",
        ),
        (
            Path("/content/BasicSR"),
            "basicsr",
            "1.4.2",
            "from .version import __gitsha__, __version__\n",
        ),
        (
            Path("/content/facexlib"),
            "facexlib",
            "0.3.0",
            "from .detection import *\nfrom .parsing import *\nfrom .utils import *\nfrom .version import __gitsha__, __version__\n",
        ),
        (
            Path("/content/GFPGAN"),
            "gfpgan",
            "1.3.8",
            "from .utils import GFPGANer\n",
        ),
    )
    for root, package_name, version, lightweight_init in source_specs:
        package_dir = root / package_name
        if not package_dir.is_dir():
            continue
        version_file = package_dir / "version.py"
        if not version_file.exists():
            version_file.write_text(
                f"__version__ = {version!r}\n__gitsha__ = 'colab-source'\nversion_info = {tuple(int(part) for part in version.split('.'))!r}\n",
                encoding="utf-8",
            )
        init_file = package_dir / "__init__.py"
        if package_name in {"realesrgan", "basicsr", "facexlib", "gfpgan"} and init_file.exists():
            # The upstream inits eagerly import training/registry modules and
            # their generated version metadata. Only inference APIs are needed
            # here, so keep the source-only package imports lightweight.
            init_file.write_text(lightweight_init, encoding="utf-8")


def _ensure_torchvision_compat() -> None:
    """Bridge a module path removed by newer torchvision releases."""

    _ensure_source_package_compat()
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
    if model_key == "photo_x2":
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )


def get_upsampler(
    mode: Any = "Auto (recommended)",
    tile_size: Any = 512,
    model_scale: Any = 4,
) -> tuple[Any, str]:
    """Load and cache a Real-ESRGAN upsampler, returning it and its model key.

    Real photos requested at 2x use the native x2 checkpoint instead of
    computing a 4x image and shrinking it. This is a substantial speed and
    memory win for the common 2x video setting.
    """

    _ensure_torchvision_compat()
    from realesrgan import RealESRGANer

    base_model_key = resolve_model(mode)
    requested_scale = _scale_value(model_scale)
    model_key = "photo_x2" if base_model_key == "photo" and requested_scale == 2 else base_model_key
    native_scale = 2 if model_key == "photo_x2" else 4
    tile = _tile_value(tile_size)
    device = _require_cuda()
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
        scale=native_scale,
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
    actual_device = next(upsampler.model.parameters()).device
    if actual_device.type != "cuda":
        raise RuntimeError(f"Real-ESRGAN loaded on {actual_device}, not on the T4 CUDA device.")
    print(f"[PARAM] Using {actual_device} · FP16={half} · {_model_label(model_key)} · tile={tile}")
    _UPSAMPLER_CACHE[cache_key] = upsampler
    return upsampler, model_key


def get_face_enhancer(mode: Any = "Auto (recommended)", tile_size: Any = 512) -> Any:
    """Load GFPGAN once, using the same tiled background upsampler."""

    _ensure_torchvision_compat()
    from gfpgan import GFPGANer

    # GFPGAN expects a 4x background upsampler; final 2x output is resized
    # after faces are pasted back.
    upsampler, model_key = get_upsampler(mode, tile_size, model_scale=4)
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
    if face_enhance:
        model_key = resolve_model(mode)
        enhancer = get_face_enhancer(mode, tile_size)
        _, _, output = enhancer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
    else:
        upsampler, model_key = get_upsampler(mode, tile_size, model_scale=scale)
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
    tile_size: Any = 512,
    face_enhance: bool = False,
) -> tuple[np.ndarray, str, int]:
    """Upscale one BGR image, retrying with a smaller tile after a CUDA OOM."""

    _require_cuda()
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    scale = _scale_value(scale)
    tile = _tile_value(tile_size)
    if tile == 0:
        # Full-frame inference is fastest for small/medium video frames. If a
        # high-resolution frame does not fit, fall back through safe tiles.
        attempts = [0, 1024, 512, 256, 128]
    else:
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
        f"{scale}x · {_model_label(model_key)}{extras} · tile {tile} · {device_text}"
    )


def upscale_image(
    input_file: Any,
    scale_choice: Any = "4x",
    model_mode: Any = "Auto (recommended)",
    face_enhance: bool = False,
    tile_size: Any = 512,
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


def _video2x_model_name(scale: int, mode: Any) -> str:
    """Choose a model bundled with Video2X 6 for the requested output."""

    # Video2X ships x2/x3/x4 AnimeVideoV3 assets. It is the only bundled
    # Video2X Real-ESRGAN family with a native x2 model.
    if scale == 2 or resolve_model(mode) == "anime":
        return "realesr-animevideov3"
    return "realesrgan-plus"


def _run_video2x(
    source: Path,
    destination: Path,
    scale: int,
    model_mode: Any,
    progress: ProgressFn = None,
) -> str:
    """Run Video2X 6's C++/Vulkan pipeline and return the model label."""

    prepare_video2x(progress)
    assert _VIDEO2X_RUNTIME is not None
    runner, cwd, env = _VIDEO2X_RUNTIME
    model_name = _video2x_model_name(scale, model_mode)
    command = [
        *runner,
        "-i",
        str(source),
        "-o",
        str(destination),
        "-p",
        "realesrgan",
        "-s",
        str(scale),
        "--realesrgan-model",
        model_name,
        "-d",
        str(_VIDEO2X_DEVICE_INDEX),
        "-c",
        "libx264",
        "-e",
        "crf=18",
        "-e",
        "preset=medium",
    ]
    _report(progress, 0.15, f"Video2X Vulkan running ({model_name}, 2x/4x)")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except OSError as error:
        raise RuntimeError(f"Could not start Video2X: {error}") from error

    # Video2X writes its progress bar with carriage returns instead of normal
    # newlines. Read the pipe without buffering it until the process exits, so
    # the Colab cell and Gradio progress indicator both stay informative.
    output_tail: list[str] = []
    text_buffer = ""
    last_cell_print = 0.0
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while selector.get_map():
            events = selector.select(timeout=0.25)
            if not events:
                continue
            for key, _ in events:
                data = os.read(key.fileobj.fileno(), 8192)
                if not data:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                text_buffer += data.decode("utf-8", errors="replace")
                parts = re.split(r"[\r\n]", text_buffer)
                text_buffer = parts.pop()
                for part in parts:
                    cleaned = part.strip()
                    if not cleaned:
                        continue
                    output_tail.append(cleaned)
                    del output_tail[:-20]
                    match = re.search(r"frame=(\d+)\s*/\s*(\d+)", cleaned)
                    if match:
                        processed, total = int(match.group(1)), int(match.group(2))
                        fraction = 0.15 + 0.80 * (processed / max(total, 1))
                        _report(progress, fraction, f"Video2X frame {processed}/{total}")
                    now = time.monotonic()
                    if now - last_cell_print >= 1.0:
                        print(f"[Video2X] {cleaned}", flush=True)
                        last_cell_print = now
        if text_buffer.strip():
            output_tail.append(text_buffer.strip())
    finally:
        selector.close()

    return_code = process.wait()
    _report(progress, 0.95, "Video2X encoding output")
    if return_code != 0 or not destination.exists() or destination.stat().st_size == 0:
        details = "\n".join(output_tail[-8:])
        raise RuntimeError(
            "Video2X failed. Confirm that Colab exposes a Vulkan GPU (the T4 should appear as device 0).\n"
            + details
        )
    _report(progress, 1.0, "Video2X finished")
    return model_name


def upscale_video(
    input_file: Any,
    scale_choice: Any = "2x",
    model_mode: Any = "Auto (recommended)",
    face_enhance: bool = False,
    tile_size: Any = 512,
    progress: ProgressFn = None,
) -> tuple[str, str]:
    """Gradio callback for videos using Video2X 6's native Vulkan pipeline."""

    source = _as_path(input_file)
    if source is None:
        raise ValueError("Upload a video first.")
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {source.suffix or 'unknown'}")
    if face_enhance:
        raise ValueError("GFPGAN is not used by Video2X. Turn off face restoration for the fast Video2X path.")
    scale = _scale_value(scale_choice)
    _tile_value(tile_size)  # Kept in the shared UI; Video2X chooses its own Vulkan tile size.

    # Read metadata only for a useful status message; Video2X owns decoding,
    # frame processing, audio copying, and final encoding.
    cap = cv2.VideoCapture(str(source))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError("Could not read the video dimensions. Try MP4 or re-encode the source first.")

    destination = _output_path(source, scale, ".mp4")
    model_name = _run_video2x(source, destination, scale, model_mode, progress)
    frame_text = f" · {total} frames" if total > 0 else ""
    status = (
        f"✅ **Done:** `{source.name}` · {width}×{height} → {width * scale}×{height * scale} · "
        f"{scale}x · Video2X {VIDEO2X_VERSION} · {model_name} · Vulkan GPU {_VIDEO2X_DEVICE_INDEX}{frame_text} · audio preserved"
    )
    return str(destination), status

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
            "T4-friendly image enhancement · Video2X Vulkan video processing · optional GFPGAN faces",
            elem_classes=["param-subtitle"],
        )
        gr.Markdown(
            f"**Runtime:** {system_info()}\n\n"
            "Images use the Python Real-ESRGAN path. Videos use Video2X 6.4.0 with ncnn/Vulkan and its bundled models. "
            "Auto is the safe photo choice; choose Anime / illustration for cartoons or line art."
        )

        with gr.Row():
            with gr.Column(scale=1):
                scale = gr.Radio(["2x", "4x"], value="2x", label="Output scale")
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
                    maximum=1024,
                    value=512,
                    step=128,
                    label="Tile size (image path)",
                    info="Used by image Real-ESRGAN. Video2X selects its own Vulkan tile size automatically.",
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
            )
            video_button = gr.Button("Upscale video", variant="primary")
            video_output = gr.Video(label="Upscaled MP4")
            video_status = gr.Markdown()
            gr.Markdown(
                f"Video is processed by **Video2X {VIDEO2X_VERSION}** using its C++/Vulkan pipeline. "
                "For 2x, the bundled `realesr-animevideov3-x2` model is used; for real-world 4x, `realesrgan-plus-x4`. "
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
    # Let Gradio choose the first free port (important when a Colab cell is rerun).
    build_app().launch(share=True, server_name="0.0.0.0")
