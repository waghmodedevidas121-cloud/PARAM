from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
import uuid
from pathlib import Path

# Colab's inline Matplotlib backend is unavailable in this standalone venv.
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

print("[startup 1/4] Importing Gradio…", flush=True)
import gradio as gr
import numpy as np
print("[startup 2/4] Importing audio, video, and GPU libraries…", flush=True)
import cv2
import librosa
import torch
from PIL import Image, ImageOps
print(f"[startup 3/4] PyTorch {torch.__version__}; CUDA={torch.cuda.is_available()}", flush=True)

# Load PyTorch's bundled CUDA/cuDNN libraries before ONNX Runtime.
try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    print("ONNX Runtime providers:", ort.get_available_providers())
except Exception as exc:
    print("ONNX Runtime preload note:", exc)

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
DATA_ROOT = CHECKPOINTS / "ditto_pytorch"
CFG_PATH = CHECKPOINTS / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl"
OUTPUT_ROOT = ROOT / "outputs" / "gradio"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)

# apt normally supplies ffmpeg. imageio-ffmpeg is a portable fallback and the
# PATH shim also covers Ditto's own final audio-mux command.
FFMPEG_BIN = shutil.which("ffmpeg")
if FFMPEG_BIN is None:
    from imageio_ffmpeg import get_ffmpeg_exe
    bundled_ffmpeg = Path(get_ffmpeg_exe()).resolve()
    shim_dir = OUTPUT_ROOT / ".bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "ffmpeg"
    if not shim.exists():
        shim.symlink_to(bundled_ffmpeg)
    os.environ["PATH"] = f"{shim_dir}:{os.environ.get('PATH', '')}"
    FFMPEG_BIN = str(shim)

# Ditto ships a tiny Cython compositor. Prefer it, but retain a vectorized
# NumPy fallback for Colab images where Python development headers are absent.
try:
    from core.utils.blend import blend_images_cy as _blend_probe
except Exception as exc:
    print("Cython compositor unavailable; using NumPy fallback:", exc)
    sys.modules.pop("core.utils.blend", None)
    blend_module = types.ModuleType("core.utils.blend")

    def blend_images_cy(mask_warped, frame_warped, frame_rgb, result):
        alpha = mask_warped[..., None].astype(np.float32, copy=False)
        blended = alpha * frame_warped + (1.0 - alpha) * frame_rgb
        result[...] = np.clip(blended, 0, 255).astype(np.uint8)

    blend_module.blend_images_cy = blend_images_cy
    sys.modules["core.utils.blend"] = blend_module

print("[startup 4/4] Building Ditto Gradio interface…", flush=True)
from inference import run, seed_everything
from stream_pipeline_offline import StreamSDK

EMOTION_MAP = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6,
    "Contempt": 7,
}
MAX_AUDIO_SECONDS = 60.0
MIN_REFERENCE_SECONDS = 2.0
MAX_REFERENCE_SECONDS = 20.0
MODEL_LOCK = threading.Lock()
_SDK: StreamSDK | None = None


def get_sdk() -> StreamSDK:
    global _SDK
    if _SDK is None:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required. In Colab select Runtime → Change runtime type → T4 GPU.")
        print("Loading Ditto PyTorch models (first request only)…")
        _SDK = StreamSDK(str(CFG_PATH), str(DATA_ROOT))
        print("Ditto models loaded.")
    return _SDK


def clean_old_jobs(max_age_hours: float = 8.0) -> None:
    cutoff = time.time() - max_age_hours * 3600
    for item in OUTPUT_ROOT.iterdir():
        try:
            if item.name.startswith("."):
                continue
            if item.is_dir() and item.stat().st_mtime < cutoff:
                shutil.rmtree(item, ignore_errors=True)
        except OSError:
            pass


def run_checked(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        tail = (result.stderr or result.stdout)[-3000:]
        raise RuntimeError(f"Command failed ({result.returncode}):\n{tail}")


def prepare_image(source_path: str, destination: Path) -> None:
    Image.MAX_IMAGE_PIXELS = 40_000_000
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        if min(width, height) < 256:
            raise ValueError("Portrait is too small. Use an image at least 256 px on each side.")
        if width * height > 40_000_000:
            raise ValueError("Portrait is too large. Use an image below 40 megapixels.")
        image.save(destination, format="PNG", optimize=True)


def prepare_reference_video(
    source_path: str,
    destination: Path,
    max_dimension: int,
) -> float:
    """Normalize a short motion reference while discarding its old audio."""
    capture = cv2.VideoCapture(source_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("Could not read the reference video. Upload a normal MP4/MOV/WebM file.")
    duration = frame_count / fps
    if duration < MIN_REFERENCE_SECONDS:
        raise ValueError(
            f"Reference video is too short ({duration:.1f}s). Use at least {MIN_REFERENCE_SECONDS:.0f}s."
        )
    if duration > MAX_REFERENCE_SECONDS:
        raise ValueError(
            f"Reference video is {duration:.1f}s. Trim it to {MAX_REFERENCE_SECONDS:.0f}s or less."
        )
    if min(width, height) < 256:
        raise ValueError("Reference video is too small. Use at least 256 px on each side.")

    scale = min(1.0, int(max_dimension) / max(width, height))
    output_width = max(2, int(width * scale) // 2 * 2)
    output_height = max(2, int(height * scale) // 2 * 2)
    run_checked([
        FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
        "-i", source_path, "-map", "0:v:0", "-an",
        "-vf", f"fps=25,scale={output_width}:{output_height}:flags=lanczos",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ])
    return duration


def prepare_audio(source_path: str, destination: Path) -> float:
    run_checked([
        FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
        "-i", source_path, "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(destination),
    ])
    duration = float(librosa.get_duration(path=str(destination)))
    if duration < 0.25:
        raise ValueError("Audio is too short; provide at least 0.25 seconds.")
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(f"Audio is {duration:.1f}s. This Colab UI limits each job to {MAX_AUDIO_SECONDS:.0f}s.")
    return duration


def add_disclosure_watermark(source: Path, destination: Path) -> bool:
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    vf = (
        "drawbox=x=w-238:y=h-52:w=226:h=40:color=black@0.55:t=fill,"
        f"drawtext=fontfile={font}:text='AI-generated avatar':"
        "fontcolor=white:fontsize=18:x=w-tw-20:y=h-th-20"
    )
    result = subprocess.run([
        FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vf", vf, "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-c:a", "copy",
        "-movflags", "+faststart", str(destination),
    ], text=True, capture_output=True)
    if result.returncode:
        print("Watermark step skipped:", result.stderr[-1200:])
        return False
    return True


def generate_avatar(
    source_video: str,
    source_image: str,
    driving_audio: str,
    emotion: str,
    sampling_steps: int,
    max_output_dimension: int,
    crop_scale: float,
    seed: int,
    add_watermark: bool,
    consent_confirmed: bool,
    progress=gr.Progress(),
):
    if not consent_confirmed:
        raise gr.Error("Confirm that you have permission to use the face, video, and audio.")
    if not source_video and not source_image:
        raise gr.Error("Upload a 2–20 second motion-reference video or one clear portrait image.")
    if not driving_audio:
        raise gr.Error("Upload or record the new driving speech audio.")

    with MODEL_LOCK:
        started = time.perf_counter()
        clean_old_jobs()
        job_dir = OUTPUT_ROOT / f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True, exist_ok=False)
        image_path = job_dir / "portrait.png"
        video_path = job_dir / "motion_reference.mp4"
        audio_path = job_dir / "speech_16k.wav"
        raw_path = job_dir / "avatar_raw.mp4"
        final_path = job_dir / "avatar.mp4"

        try:
            progress(0.03, desc="Validating motion reference and audio…")
            reference_duration = None
            if source_video:
                reference_duration = prepare_reference_video(
                    source_video, video_path, int(max_output_dimension)
                )
                source_path = video_path
                mode_name = "Video Twin (reference motion)"
            else:
                prepare_image(source_image, image_path)
                source_path = image_path
                mode_name = "Photo Avatar (face/head motion)"
            duration = prepare_audio(driving_audio, audio_path)

            progress(0.10, desc="Loading models (first job takes longer)…")
            sdk = get_sdk()
            seed = max(0, int(seed))
            seed_everything(seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            progress(0.18, desc="Animating the avatar…")
            more_kwargs = {
                "setup_kwargs": {
                    "emo": EMOTION_MAP.get(emotion, 4),
                    "sampling_timesteps": int(sampling_steps),
                    "max_size": int(max_output_dimension),
                    "crop_scale": float(crop_scale),
                },
                "run_kwargs": {},
            }
            run(sdk, str(audio_path), str(source_path), str(raw_path), more_kwargs)
            if not raw_path.exists() or raw_path.stat().st_size == 0:
                raise RuntimeError("Ditto finished without creating an output video.")

            progress(0.92, desc="Finalizing video…")
            watermarked = False
            if add_watermark:
                watermarked = add_disclosure_watermark(raw_path, final_path)
            output_path = final_path if watermarked else raw_path

            elapsed = time.perf_counter() - started
            peak = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available() else 0.0
            )
            progress(1.0, desc="Done")
            status = (
                f"### ✅ Avatar ready\n"
                f"- Mode: **{mode_name}**\n"
                f"- New audio: **{duration:.1f}s** · Render: **{elapsed:.1f}s**\n"
                f"- Emotion: **{emotion}** · Seed: **{seed}** · Steps: **{int(sampling_steps)}**\n"
                f"- PyTorch peak VRAM: **{peak:.2f} GiB**"
            )
            if reference_duration is not None:
                status += f"\n- Motion reference: **{reference_duration:.1f}s**, mirrored/looped to match the audio"
            if add_watermark and not watermarked:
                status += "\n- ⚠️ Disclosure watermark could not be added; the raw result is shown."
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return str(output_path), status
        except gr.Error:
            raise
        except Exception as exc:
            traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error(
                f"Generation failed: {exc}. If a worker error occurred, restart the Launch cell and try a frontal portrait."
            ) from exc


CSS = """
.gradio-container {max-width: 1180px !important;}
.hero {text-align:center; padding: 12px 0 4px;}
.hero h1 {font-size: 2.25rem; margin-bottom: .25rem;}
.notice {border-left: 4px solid #6366f1; padding: 10px 14px; background: rgba(99,102,241,.08); border-radius: 8px;}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet"),
    css=CSS,
    title="Open Avatar Studio — Video Twin",
) as demo:
    gr.HTML("""
    <div class="hero">
      <h1>🎭 Open Avatar Studio</h1>
      <p>Reference body motion + new speech → a T4-friendly Video Twin</p>
    </div>
    """)
    gr.Markdown(
        "<div class='notice'><b>Video Twin mode (Avatar V‑lite):</b> upload a short video "
        "where you naturally gesture. Ditto preserves those real body, hand, clothing, and "
        "background frames while replacing facial speech motion and audio. The motion is "
        "mirrored/looped; unlike proprietary HeyGen Avatar V, it is not regenerated to "
        "understand each new sentence.</div>"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Tabs():
                with gr.Tab("🎬 Video Twin · full-frame motion", id="video-twin"):
                    source_video = gr.Video(
                        label="1 · Motion-reference video (2–20 seconds)",
                        sources=["upload"],
                        height=360,
                    )
                    gr.Markdown(
                        "**Recommended:** 10–15 seconds, one visible person, front-facing, "
                        "waist-up/full-body, natural hand gestures, static camera, and no cuts. "
                        "The original video's audio is discarded."
                    )
                with gr.Tab("🖼️ Photo Avatar · face/head only", id="photo-avatar"):
                    source_image = gr.Image(
                        label="Alternative portrait image",
                        type="filepath",
                        sources=["upload", "webcam"],
                        height=360,
                    )
                    gr.Markdown(
                        "Photo mode remains available, but it cannot invent full-body movement. "
                        "If both inputs are supplied, **Video Twin takes priority**."
                    )

            driving_audio = gr.Audio(
                label="2 · New driving speech (max 60 seconds)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            emotion = gr.Dropdown(
                choices=list(EMOTION_MAP), value="Neutral", label="3 · Facial expression style"
            )

            with gr.Accordion("Advanced controls", open=False):
                sampling_steps = gr.Slider(
                    10, 50, value=20, step=10,
                    label="Motion sampling steps",
                    info="10 is faster; 30–50 may improve difficult clips.",
                )
                max_output_dimension = gr.Radio(
                    choices=[
                        ("Recommended T4 · 720 px", 720),
                        ("High-RAM · 1080 px", 1080),
                        ("Source up to 1920 px", 1920),
                    ],
                    value=720,
                    label="Maximum output dimension",
                )
                crop_scale = gr.Slider(
                    1.8, 3.2, value=2.3, step=0.1,
                    label="Face crop scale",
                    info="Increase if the head is clipped; 2.3 is the model default.",
                )
                seed = gr.Number(value=1024, precision=0, label="Seed")
                add_watermark = gr.Checkbox(
                    value=True, label="Add ‘AI-generated avatar’ disclosure watermark"
                )

            consent = gr.Checkbox(
                value=False,
                label=(
                    "I own or have explicit permission to use this person's face, reference "
                    "video, and audio, and I will disclose synthetic media where appropriate."
                ),
            )
            generate_button = gr.Button("✨ Generate Video Twin", variant="primary", size="lg")

            example_image = ROOT / "example" / "image.png"
            example_audio = ROOT / "example" / "audio.wav"
            if example_image.exists() and example_audio.exists():
                gr.Examples(
                    examples=[[str(example_image), str(example_audio)]],
                    inputs=[source_image, driving_audio],
                    label="Official photo-mode sample",
                )

        with gr.Column(scale=6):
            output_video = gr.Video(label="Generated avatar", height=560)
            status = gr.Markdown(
                "For body/hand motion, upload a reference video and new speech, confirm permission, then generate."
            )

    generate_button.click(
        fn=generate_avatar,
        inputs=[
            source_video,
            source_image,
            driving_audio,
            emotion,
            sampling_steps,
            max_output_dimension,
            crop_scale,
            seed,
            add_watermark,
            consent,
        ],
        outputs=[output_video, status],
        api_name=False,
    )

    gr.Markdown(
        "---\n**Video Twin input tips:** use one person, a locked camera, visible hands, "
        "even lighting, direct eye contact, and gestures that return to a neutral pose. "
        "Avoid cuts, face occlusion, fast turns, and hands crossing the face. This T4 mode "
        "reuses recorded motion—it does not learn a persistent identity or create "
        "script-aware gestures like the proprietary Avatar V service. "
        "Do not use it for impersonation, fraud, harassment, or non-consensual media. "
        "[Ditto source](https://github.com/antgroup/ditto-talkinghead) · "
        "[Apache‑2.0 license](https://github.com/antgroup/ditto-talkinghead/blob/main/LICENSE)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.queue(max_size=4, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(OUTPUT_ROOT), str(ROOT / "example")],
    )


if __name__ == "__main__":
    main()
