from __future__ import annotations

import argparse
import gc
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

# Colab exports an inline Matplotlib backend that is unavailable in the venv.
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

print("[startup 1/3] Importing Gradio and media libraries…", flush=True)
import gradio as gr
import soundfile as sf
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "inference_t4.yaml"
OUTPUT_ROOT = ROOT / "outputs" / "omniavatar_gradio"
DEMO_ROOT = ROOT / "demo_out"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)

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

TORCHRUN = Path(sys.executable).parent / "torchrun"
MAX_AUDIO_SECONDS = 8.0
MODEL_LOCK = threading.Lock()

DEFAULT_NEGATIVE = (
    "Vivid color tones, background or camera moving quickly, screen switching, subtitles, "
    "special effects, mutation, overexposed, static, blurred details, painting, still image, "
    "worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn "
    "hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, motionless "
    "image, chaotic or crowded background, extra limbs, walking backward"
)

PROMPT_PRESETS = {
    "Natural presenter": (
        "A realistic waist-up video of a confident presenter speaking directly to the camera. "
        "The presenter uses natural, dynamic, rhythmic hand gestures that complement the speech. "
        "Both hands remain clearly visible and unobstructed. Facial expressions are expressive "
        "and emotionally appropriate. The camera is locked and steady, with soft studio lighting, "
        "sharp details, realistic skin, and a clean professional background."
    ),
    "Calm instructor": (
        "A realistic waist-up instructor speaking calmly to the camera, using subtle open-palm "
        "hand gestures and occasional natural nods. The delivery is warm and trustworthy. Hands "
        "stay visible below the face. Locked camera, even studio light, clean background, sharp detail."
    ),
    "Energetic creator": (
        "A realistic social-media presenter speaking energetically to the camera with lively but "
        "controlled hand gestures, expressive eyebrows, smiles, and natural upper-body movement. "
        "Hands are fully visible and never cover the face. Locked camera, bright clean studio, sharp detail."
    ),
    "Storyteller": (
        "A realistic waist-up storyteller speaking directly to the camera with emotionally varied "
        "expressions, gentle head turns, and meaningful hand gestures timed to the delivery. The "
        "camera remains stationary. Cinematic soft lighting, stable clean background, realistic detail."
    ),
}


def clean_old_jobs(max_age_hours: float = 10.0) -> None:
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
        detail = (result.stderr or result.stdout)[-3000:]
        raise RuntimeError(f"Command failed ({result.returncode}):\n{detail}")


def prepare_image(source_path: str, destination: Path) -> None:
    Image.MAX_IMAGE_PIXELS = 40_000_000
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        if min(width, height) < 384:
            raise ValueError("Image is too small. Use at least 384 px on each side.")
        if width * height > 40_000_000:
            raise ValueError("Image is too large. Use an image below 40 megapixels.")
        image.save(destination, format="JPEG", quality=95, subsampling=0)


def prepare_audio(source_path: str, destination: Path) -> float:
    run_checked([
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ])
    audio_info = sf.info(str(destination))
    duration = float(audio_info.frames / audio_info.samplerate)
    if duration < 0.5:
        raise ValueError("Audio is too short. Use at least 0.5 seconds.")
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio is {duration:.1f}s. This T4 notebook limits a job to {MAX_AUDIO_SECONDS:.0f}s "
            "because generation is extremely slow. Trim the audio and combine clips later."
        )
    return duration


def add_watermark(source: Path, destination: Path) -> bool:
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    vf = (
        "drawbox=x=w-238:y=h-52:w=226:h=40:color=black@0.55:t=fill,"
        f"drawtext=fontfile={font}:text='AI-generated avatar':"
        "fontcolor=white:fontsize=18:x=w-tw-20:y=h-th-20"
    )
    result = subprocess.run([
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ], text=True, capture_output=True)
    if result.returncode:
        print("Watermark step skipped:", result.stderr[-1200:], flush=True)
        return False
    return True


def newest_result(previous: set[Path], started: float) -> Path | None:
    candidates = list(DEMO_ROOT.rglob("result_000_000_wav.mp4")) if DEMO_ROOT.exists() else []
    candidates = [
        path for path in candidates
        if path not in previous and path.stat().st_mtime >= started - 3
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def tail_text(path: Path, max_chars: int = 6000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return "No generation log was available."


def generate_avatar(
    image_path: str,
    audio_path: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    prompt_cfg: float,
    audio_cfg: float,
    max_tokens: int,
    tea_cache: float,
    seed: int,
    disclosure_watermark: bool,
    consent_confirmed: bool,
    progress=gr.Progress(),
):
    if not consent_confirmed:
        raise gr.Error("Confirm that you have permission to use the image and audio.")
    if not image_path:
        raise gr.Error("Upload one clear waist-up or full-body image.")
    if not audio_path:
        raise gr.Error("Upload or record speech audio.")
    prompt = " ".join((prompt or "").replace("@@", " ").split())
    negative_prompt = " ".join((negative_prompt or DEFAULT_NEGATIVE).replace("@@", " ").split())
    if len(prompt) < 20:
        raise gr.Error("Add a descriptive motion prompt of at least 20 characters.")

    with MODEL_LOCK:
        started_clock = time.perf_counter()
        started_epoch = time.time()
        clean_old_jobs()
        job_dir = OUTPUT_ROOT / f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True, exist_ok=False)
        input_image = job_dir / "source.jpg"
        input_audio = job_dir / "speech.wav"
        input_txt = job_dir / "input.txt"
        job_config = job_dir / "inference.yaml"
        log_path = job_dir / "generation.log"
        raw_output = job_dir / "omniavatar_raw.mp4"
        final_output = job_dir / "omniavatar.mp4"

        process: subprocess.Popen | None = None
        try:
            progress(0.03, desc="Validating inputs…")
            prepare_image(image_path, input_image)
            duration = prepare_audio(audio_path, input_audio)
            input_txt.write_text(
                f"{prompt}@@{input_image}@@{input_audio}\n", encoding="utf-8"
            )

            prior_results = set(DEMO_ROOT.rglob("result_000_000_wav.mp4")) if DEMO_ROOT.exists() else set()
            with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
            config.update({
                "seed": max(0, int(seed)),
                "num_steps": int(steps),
                "guidance_scale": float(prompt_cfg),
                "audio_scale": float(audio_cfg),
                "max_tokens": int(max_tokens),
                "tea_cache_l1_thresh": float(tea_cache),
                "num_persistent_param_in_dit": 7000000000,
                "overlap_frame": 13,
                "negative_prompt": negative_prompt,
            })
            job_config.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

            master_port = random.randint(29600, 29990)
            command = [
                str(TORCHRUN),
                "--standalone",
                "--nproc_per_node=1",
                f"--master_port={master_port}",
                "scripts/inference.py",
                "--config",
                str(job_config),
                "--input_file",
                str(input_txt),
            ]
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": "0",
                "MPLBACKEND": "Agg",
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "2",
            })

            progress(0.08, desc="Loading ~18 GB of model weights…")
            print("\n[OmniAvatar job]", " ".join(command), flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                    log_file.flush()
                return_code = process.wait()

            if return_code != 0:
                detail = tail_text(log_path)
                if return_code in (-9, 137) or "SIGKILL" in detail:
                    raise RuntimeError(
                        "The process was killed by host RAM pressure. Select a High-RAM runtime "
                        "or use a runtime with at least 16–24 GB system RAM.\n" + detail[-1800:]
                    )
                raise RuntimeError(f"OmniAvatar exited with code {return_code}.\n{detail[-2400:]}")

            progress(0.92, desc="Collecting and finalizing the MP4…")
            result = newest_result(prior_results, started_epoch)
            if result is None or result.stat().st_size == 0:
                raise RuntimeError(
                    "Inference completed but its result MP4 was not found.\n" + tail_text(log_path)[-2400:]
                )
            shutil.copy2(result, raw_output)
            watermarked = disclosure_watermark and add_watermark(raw_output, final_output)
            selected_output = final_output if watermarked else raw_output

            elapsed = time.perf_counter() - started_clock
            status = (
                "### ✅ OmniAvatar video ready\n"
                f"- Input audio: **{duration:.1f}s** · Total job time: **{elapsed / 60:.1f} min**\n"
                f"- Steps: **{int(steps)}** · Prompt CFG: **{float(prompt_cfg):.1f}** · "
                f"Audio CFG: **{float(audio_cfg):.1f}**\n"
                f"- Token budget: **{int(max_tokens)}** · TeaCache: **{float(tea_cache):.2f}**"
            )
            if disclosure_watermark and not watermarked:
                status += "\n- ⚠️ Watermarking failed; the raw generated result is shown."

            # Keep the user-facing copy and log, remove the duplicate dynamic CLI folder.
            dynamic_dir = result.parent
            if DEMO_ROOT in dynamic_dir.parents:
                shutil.rmtree(dynamic_dir, ignore_errors=True)
            gc.collect()
            progress(1.0, desc="Done")
            return str(selected_output), status, str(log_path)
        except gr.Error:
            raise
        except Exception as exc:
            traceback.print_exc()
            if process is not None and process.poll() is None:
                process.terminate()
            raise gr.Error(f"Generation failed: {exc}") from exc


CSS = """
.gradio-container {max-width: 1220px !important;}
.hero {text-align:center; padding: 12px 0 4px;}
.hero h1 {font-size:2.25rem; margin-bottom:.25rem;}
.notice {border-left:4px solid #7c3aed; padding:12px 15px; background:rgba(124,58,237,.09); border-radius:8px;}
.warning {border-left:4px solid #f59e0b; padding:10px 14px; background:rgba(245,158,11,.09); border-radius:8px;}
"""

print("[startup 2/3] Building OmniAvatar interface…", flush=True)
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo"),
    css=CSS,
    title="OmniAvatar 1.3B — T4 Studio",
) as demo:
    gr.HTML("""
    <div class="hero">
      <h1>🧑‍🎤 OmniAvatar 1.3B Studio</h1>
      <p>Photo + speech + behavior prompt → generative face and adaptive body animation</p>
    </div>
    """)
    gr.Markdown(
        "<div class='notice'><b>True generative mode:</b> unlike Video Twin replay, OmniAvatar "
        "generates new face, head, shoulder, hand, and body motion guided by the audio and prompt.</div>"
    )
    gr.Markdown(
        "<div class='warning'><b>T4 reality:</b> this model is extremely slow. Start with 2–4 seconds "
        "of audio, 10–20 steps, and the Safe token profile. A 6-second clip can take tens of minutes "
        "or longer. At least 16 GB system RAM is strongly recommended.</div>"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            source_image = gr.Image(
                label="1 · Source image (waist-up/full-body recommended)",
                type="filepath",
                sources=["upload", "webcam"],
                height=410,
            )
            driving_audio = gr.Audio(
                label="2 · Speech audio (0.5–8 seconds; start with 2–4s)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            preset = gr.Dropdown(
                choices=list(PROMPT_PRESETS),
                value="Natural presenter",
                label="3 · Motion-prompt preset",
            )
            prompt = gr.Textbox(
                label="Behavior and scene prompt",
                value=PROMPT_PRESETS["Natural presenter"],
                lines=5,
            )
            preset.change(
                fn=lambda name: PROMPT_PRESETS.get(name, PROMPT_PRESETS["Natural presenter"]),
                inputs=preset,
                outputs=prompt,
                queue=False,
            )

            with gr.Accordion("Advanced generation controls", open=False):
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    value=DEFAULT_NEGATIVE,
                    lines=4,
                )
                steps = gr.Slider(
                    10, 30, value=20, step=5,
                    label="Sampling steps",
                    info="10 is experimental/fast; official guidance is 20–50.",
                )
                prompt_cfg = gr.Slider(3.0, 6.0, value=4.5, step=0.1, label="Prompt CFG")
                audio_cfg = gr.Slider(3.0, 7.0, value=5.0, step=0.1, label="Audio CFG")
                max_tokens = gr.Radio(
                    choices=[
                        ("T4 Safe · 12k (slowest, lowest peak)", 12000),
                        ("T4 Balanced · 16k", 16000),
                        ("Faster · 24k (more VRAM)", 24000),
                        ("Official · 30k (highest peak)", 30000),
                    ],
                    value=16000,
                    label="Per-segment token budget",
                )
                tea_cache = gr.Slider(
                    0.0, 0.10, value=0.0, step=0.01,
                    label="TeaCache acceleration",
                    info="0 gives best quality. Higher is faster but can visibly reduce motion quality.",
                )
                seed = gr.Number(value=42, precision=0, label="Seed")
                disclosure_watermark = gr.Checkbox(
                    value=True, label="Add ‘AI-generated avatar’ disclosure watermark"
                )

            consent = gr.Checkbox(
                value=False,
                label=(
                    "I own or have explicit permission to use this face and audio, and I will "
                    "disclose synthetic media where appropriate."
                ),
            )
            generate_button = gr.Button("✨ Generate with OmniAvatar", variant="primary", size="lg")

            example_image = ROOT / "examples" / "images" / "0000.jpeg"
            example_audio = ROOT / "examples" / "audios" / "0000.MP3"
            if example_image.exists() and example_audio.exists():
                gr.Examples(
                    examples=[[str(example_image), str(example_audio)]],
                    inputs=[source_image, driving_audio],
                    label="Official OmniAvatar sample inputs",
                )

        with gr.Column(scale=6):
            output_video = gr.Video(label="Generated OmniAvatar video", height=560)
            status = gr.Markdown("Use a short test clip first, then generate.")
            generation_log = gr.File(label="Generation log")

    generate_button.click(
        fn=generate_avatar,
        inputs=[
            source_image,
            driving_audio,
            prompt,
            negative_prompt,
            steps,
            prompt_cfg,
            audio_cfg,
            max_tokens,
            tea_cache,
            seed,
            disclosure_watermark,
            consent,
        ],
        outputs=[output_video, status, generation_log],
        api_name=False,
    )

    gr.Markdown(
        "---\n**Input tips:** use one front-facing person with visible shoulders and hands, a simple "
        "background, and clean speech. Prompt the desired behavior explicitly. Avoid crowded scenes, "
        "occluded hands, extreme profiles, and long first tests. The 1.3B checkpoint is a lightweight "
        "research model; its quality is below the 14B version. "
        "[Official OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) · "
        "[Apache‑2.0 license](https://github.com/Omni-Avatar/OmniAvatar/blob/main/LICENSE.txt)"
    )

print("[startup 3/3] Interface ready; starting Gradio…", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.queue(max_size=2, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(OUTPUT_ROOT), str(ROOT / "examples")],
    )


if __name__ == "__main__":
    main()
