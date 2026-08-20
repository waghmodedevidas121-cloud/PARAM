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
import uuid
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

print("[startup 1/3] Importing Gradio and media helpers…", flush=True)
import gradio as gr
import soundfile as sf
from PIL import Image, ImageOps

APP_ROOT = Path(__file__).resolve().parent
SAD_ROOT = Path("/content/SadTalker")
LATENT_ROOT = Path("/content/LatentSync")
SAD_PYTHON = Path("/content/sadtalker-env/bin/python")
LATENT_PYTHON = Path("/content/latentsync-env/bin/python")
OUTPUT_ROOT = APP_ROOT / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = shutil.which("ffmpeg")
if FFMPEG_BIN is None:
    from imageio_ffmpeg import get_ffmpeg_exe
    FFMPEG_BIN = get_ffmpeg_exe()

MODEL_LOCK = threading.Lock()
MAX_AUDIO_SECONDS = 30.0


def clean_old_jobs(max_age_hours: float = 10.0) -> None:
    cutoff = time.time() - max_age_hours * 3600
    for item in OUTPUT_ROOT.iterdir():
        try:
            if item.is_dir() and item.stat().st_mtime < cutoff:
                shutil.rmtree(item, ignore_errors=True)
        except OSError:
            pass


def run_checked(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])


def prepare_image(source: str, destination: Path) -> None:
    Image.MAX_IMAGE_PIXELS = 40_000_000
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        if min(width, height) < 256:
            raise ValueError("Image is too small. Use at least 256 px on each side.")
        image.save(destination, "PNG", optimize=True)


def prepare_audio(source: str, destination: Path) -> float:
    run_checked([
        FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
        "-i", source, "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(destination),
    ])
    info = sf.info(str(destination))
    duration = float(info.frames / info.samplerate)
    if duration < 0.5:
        raise ValueError("Audio is too short. Use at least 0.5 seconds.")
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(f"Audio is {duration:.1f}s; trim it to {MAX_AUDIO_SECONDS:.0f}s or less.")
    return duration


def prepare_video(source: str, destination: Path) -> None:
    run_checked([
        FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
        "-i", source, "-an", "-vf", "fps=25",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ])


def stream_process(command: list[str], cwd: Path, env: dict[str, str], log_file) -> int:
    print("\n[run]", " ".join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    return process.wait()


def add_watermark(source: Path, destination: Path) -> bool:
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
    return result.returncode == 0


def generate(
    mode: str,
    image: str,
    source_video: str,
    audio: str,
    preprocess: str,
    pose_style: int,
    expression_scale: float,
    stable_head: bool,
    latent_steps: int,
    latent_guidance: float,
    deepcache: bool,
    seed: int,
    disclosure: bool,
    consent: bool,
    progress=gr.Progress(),
):
    if not consent:
        raise gr.Error("Confirm that you have permission to use the face, video, and audio.")
    if not audio:
        raise gr.Error("Upload or record driving speech audio.")
    if mode != "LatentSync only (video + audio)" and not image:
        raise gr.Error("Upload a source portrait for SadTalker.")
    if mode == "LatentSync only (video + audio)" and not source_video:
        raise gr.Error("Upload an existing source video for LatentSync-only mode.")

    with MODEL_LOCK:
        clean_old_jobs()
        started = time.perf_counter()
        job = OUTPUT_ROOT / f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job.mkdir(parents=True, exist_ok=False)
        input_image = job / "source.png"
        input_audio = job / "speech.wav"
        input_video = job / "source_video.mp4"
        sad_dir = job / "sadtalker_results"
        sad_video = job / "sadtalker.mp4"
        synced_video = job / "latentsync.mp4"
        final_video = job / "avatar.mp4"
        log_path = job / "pipeline.log"
        sad_seconds = 0.0
        latent_seconds = 0.0

        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0",
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        })
        # Torch 2.0 in SadTalker rejects the newer expandable_segments option.
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        try:
            progress(0.03, desc="Preparing inputs…")
            duration = prepare_audio(audio, input_audio)
            if image:
                prepare_image(image, input_image)
            if source_video:
                prepare_video(source_video, input_video)

            with log_path.open("w", encoding="utf-8") as log:
                if mode != "LatentSync only (video + audio)":
                    progress(0.08, desc="SadTalker: generating head, eye, and expression motion…")
                    sad_dir.mkdir(parents=True, exist_ok=True)
                    sad_command = [
                        str(SAD_PYTHON), "inference.py",
                        "--driven_audio", str(input_audio),
                        "--source_image", str(input_image),
                        "--checkpoint_dir", "checkpoints",
                        "--result_dir", str(sad_dir),
                        "--preprocess", preprocess,
                        "--size", "256",
                        "--pose_style", str(int(pose_style)),
                        "--expression_scale", str(float(expression_scale)),
                        "--batch_size", "2",
                    ]
                    if stable_head:
                        sad_command.append("--still")
                    stage_start = time.perf_counter()
                    code = stream_process(sad_command, SAD_ROOT, env, log)
                    sad_seconds = time.perf_counter() - stage_start
                    if code:
                        raise RuntimeError(f"SadTalker exited with code {code}")
                    candidates = sorted(sad_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
                    if not candidates:
                        raise RuntimeError("SadTalker completed but no MP4 was found.")
                    shutil.copy2(candidates[-1], sad_video)
                    latent_input = sad_video
                else:
                    latent_input = input_video

                if mode != "SadTalker only (fast draft)":
                    progress(0.48, desc="LatentSync 1.5: refining final lip synchronization…")
                    temp_dir = job / "latentsync_temp"
                    latent_command = [
                        str(LATENT_PYTHON), "-m", "scripts.inference",
                        "--unet_config_path", "configs/unet/stage2.yaml",
                        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
                        "--video_path", str(latent_input),
                        "--audio_path", str(input_audio),
                        "--video_out_path", str(synced_video),
                        "--inference_steps", str(int(latent_steps)),
                        "--guidance_scale", str(float(latent_guidance)),
                        "--seed", str(int(seed)),
                        "--temp_dir", str(temp_dir),
                    ]
                    if deepcache:
                        latent_command.append("--enable_deepcache")
                    latent_env = env.copy()
                    # Supported by LatentSync's Torch 2.5, but not SadTalker's Torch 2.0.
                    latent_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                    stage_start = time.perf_counter()
                    code = stream_process(latent_command, LATENT_ROOT, latent_env, log)
                    latent_seconds = time.perf_counter() - stage_start
                    if code:
                        raise RuntimeError(f"LatentSync exited with code {code}")
                    if not synced_video.is_file():
                        raise RuntimeError("LatentSync completed but no MP4 was found.")
                    raw_output = synced_video
                else:
                    raw_output = sad_video

            progress(0.95, desc="Finalizing video…")
            watermarked = disclosure and add_watermark(raw_output, final_video)
            selected = final_video if watermarked else raw_output
            elapsed = time.perf_counter() - started
            status = (
                "### ✅ Avatar ready\n"
                f"- Mode: **{mode}** · Audio: **{duration:.1f}s**\n"
                f"- SadTalker: **{sad_seconds:.1f}s** · LatentSync: **{latent_seconds:.1f}s**\n"
                f"- Total: **{elapsed:.1f}s** · Latent steps: **{int(latent_steps)}**"
            )
            if disclosure and not watermarked:
                status += "\n- ⚠️ Disclosure watermark could not be added; raw output is shown."
            gc.collect()
            progress(1.0, desc="Done")
            return str(selected), status, str(log_path)
        except gr.Error:
            raise
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(f"Pipeline failed: {exc}. Download/check the pipeline log if available.") from exc


CSS = """
.gradio-container {max-width:1220px !important;}
.hero {text-align:center; padding:12px 0 4px;}
.hero h1 {font-size:2.2rem; margin-bottom:.25rem;}
.notice {border-left:4px solid #2563eb; padding:11px 15px; background:rgba(37,99,235,.08); border-radius:8px;}
"""

print("[startup 2/3] Building SadTalker + LatentSync interface…", flush=True)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="violet"), css=CSS, title="SadTalker + LatentSync") as demo:
    gr.HTML("""
    <div class="hero"><h1>🎙️ SadTalker + LatentSync 1.5</h1>
    <p>Single photo motion generation followed by high-accuracy lip refinement</p></div>
    """)
    gr.Markdown(
        "<div class='notice'><b>Recommended combined mode:</b> SadTalker first creates head, eye, "
        "and expression motion from one photo; LatentSync 1.5 then regenerates the mouth for stronger "
        "audio alignment and cleaner temporal lip motion. T4-compatible.</div>"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            mode = gr.Radio(
                choices=[
                    "SadTalker + LatentSync (recommended)",
                    "SadTalker only (fast draft)",
                    "LatentSync only (video + audio)",
                ],
                value="SadTalker + LatentSync (recommended)",
                label="Pipeline mode",
            )
            image = gr.Image(label="Source portrait (required for SadTalker)", type="filepath", sources=["upload", "webcam"], height=360)
            source_video = gr.Video(label="Existing video (only for LatentSync-only mode)", sources=["upload"], height=220)
            audio = gr.Audio(label="Driving speech (max 30 seconds)", type="filepath", sources=["upload", "microphone"])

            with gr.Accordion("SadTalker motion controls", open=True):
                preprocess = gr.Radio(
                    choices=[("Full image / body stays visible", "full"), ("Face crop", "crop")],
                    value="full", label="Framing",
                )
                pose_style = gr.Slider(0, 45, value=0, step=1, label="Head-pose style")
                expression_scale = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="Expression scale")
                stable_head = gr.Checkbox(value=False, label="More stable head (less pose motion)")

            with gr.Accordion("LatentSync quality controls", open=True):
                latent_steps = gr.Slider(10, 30, value=20, step=2, label="Inference steps")
                latent_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Guidance scale")
                deepcache = gr.Checkbox(value=True, label="Enable DeepCache acceleration")
                seed = gr.Number(value=1247, precision=0, label="Seed")
                disclosure = gr.Checkbox(value=True, label="Add AI-generated disclosure watermark")

            consent = gr.Checkbox(
                value=False,
                label="I own or have permission to use this face/video/audio and will disclose synthetic media.",
            )
            generate_button = gr.Button("✨ Generate and refine lips", variant="primary", size="lg")

        with gr.Column(scale=6):
            output = gr.Video(label="Final video", height=560)
            status = gr.Markdown("Upload inputs and generate.")
            pipeline_log = gr.File(label="Pipeline log")

    generate_button.click(
        fn=generate,
        inputs=[mode, image, source_video, audio, preprocess, pose_style, expression_scale, stable_head,
                latent_steps, latent_guidance, deepcache, seed, disclosure, consent],
        outputs=[output, status, pipeline_log],
        api_name=False,
    )

    gr.Markdown(
        "---\n**Tips:** start with a front-facing high-resolution portrait and 3–8 seconds of clean speech. "
        "Full mode preserves the original body/background but only the head region moves. LatentSync 1.5 "
        "needs about 8 GB VRAM; 1.6 is intentionally excluded because it needs about 18 GB. "
        "[SadTalker](https://github.com/OpenTalker/SadTalker) · "
        "[LatentSync](https://github.com/bytedance/LatentSync)"
    )

print("[startup 3/3] Interface ready; starting Gradio…", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.queue(max_size=3, default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        show_error=True, allowed_paths=[str(OUTPUT_ROOT)],
    )


if __name__ == "__main__":
    main()
