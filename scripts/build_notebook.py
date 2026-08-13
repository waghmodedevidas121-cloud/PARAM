#!/usr/bin/env python3
"""Generate Voicebox_Colab.ipynb."""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "Voicebox_Colab.ipynb"


def _package_b64() -> str:
    """Embed the Python package so the notebook runs without a GitHub clone.

    The repo is private and the session branch name contains a slash
    (`arena/019ffae2-param`). GitHub/Colab often split that into
    ref=`arena` + path=`019ffae2-param`, which 404s. Shipping the
    package inside the notebook avoids that fetch entirely.
    """
    buf = io.BytesIO()
    include = [
        "voicebox_colab",
        "launch.py",
        "requirements-colab.txt",
        "LICENSE",
        "README.md",
        "pyproject.toml",
    ]
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in include:
            path = ROOT / name
            if path.exists():
                tar.add(path, arcname=name, filter=_tar_filter)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = info.name.replace("\\", "/")
    if "__pycache__" in name.split("/") or name.endswith(".pyc"):
        return None
    return info


def md(source: str) -> dict:
    lines = source.strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in lines.split("\n")],
    }


def code(source: str) -> dict:
    lines = source.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln + "\n" for ln in lines.split("\n")],
    }


cells = [
    md(
        """# Voicebox Colab

**The open-source AI voice studio — on Google Colab.**

This notebook is a Colab + Gradio adaptation of
[jamiepine/voicebox](https://github.com/jamiepine/voicebox) (`main`).
It is **not** a standalone Chatterbox demo.

Kept from Voicebox:

- seven TTS engines and the `ModelConfig` registry
- voice profiles (`cloned` / `preset`)
- sentence-boundary chunking + crossfade
- pedalboard effects (same 8 DSP units + 4 presets)
- honest per-engine capabilities (no fake emotion sliders)

Replaced:

- Tauri / React desktop UI → **Gradio**
- OS app-data + SQLite → `/content/voicebox_colab/`

> **Runtime → Change runtime type → GPU** (T4 is enough for any *single* engine).
"""
    ),
    md(
        """## Architecture (Voicebox `main` → Colab)

```
Google Colab + CUDA
        │
        ├── voicebox_colab/                  # port of backend/
        │     ├── backends/                  # qwen, customvoice, chatterbox,
        │     │                              # turbo, kokoro, luxtts, tada
        │     ├── chunked_tts.py             # backend/utils/chunked_tts.py
        │     ├── effects.py                 # backend/utils/effects.py
        │     ├── profiles.py                # simplified services/profiles.py
        │     └── services/model_manager.py  # one model loaded at a time
        │
        └── Gradio studio
              Studio · Voices · Models · History · About
```

Official VRAM (Voicebox `docs/content/docs/developer/model-management.mdx`):

| Model | VRAM | Notes |
|---|---|---|
| Kokoro 82M | ~0.15 GB | 50 preset voices, CPU realtime |
| LuxTTS | ~1 GB | English cloning, 48 kHz |
| Chatterbox Turbo | ~1.5 GB | English + `[laugh]` `[sigh]` tags |
| Qwen / CustomVoice 0.6B | ~2 GB | 10 languages |
| Chatterbox Multilingual | ~3 GB | 23 languages, Hindi, Hebrew, … |
| TADA 1B | ~4 GB | English, long coherent audio |
| Qwen / CustomVoice 1.7B | ~6 GB | highest-quality Qwen |
| TADA 3B-ML | ~8 GB | 10 languages |

**Do not load two engines at once.** Use **Models → Unload / Clear GPU**.
"""
    ),
    md("## 1. GPU check"),
    code(
        """import subprocess, sys

print("Python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: no GPU. Enable a GPU runtime or stick to Kokoro / LuxTTS on CPU.")
except ImportError:
    print("torch not imported yet (Colab usually has it).")

!nvidia-smi || true
"""
    ),
    md(
        """## 2. Install the Voicebox Colab package

This cell does **not** fetch GitHub. The `voicebox_colab` package is
embedded in the notebook so a private repo or a slash in the branch name
(`arena/019ffae2-param`) cannot break setup.

If `voicebox_colab/` is already next to the notebook (full repo upload /
Drive mount), that copy is used instead.
"""
    ),
    code(
        r'''import os, sys, io, tarfile, base64
from pathlib import Path

ROOT_CANDIDATES = [
    Path.cwd(),
    Path("/content/PARAM"),
    Path("/content"),
]

def find_pkg():
    for root in ROOT_CANDIDATES:
        if (root / "voicebox_colab" / "ui" / "gradio_app.py").exists():
            return root
    return None

root = find_pkg()
if root is None:
    dest = Path("/content/PARAM") if Path("/content").exists() else Path.cwd() / "PARAM"
    dest.mkdir(parents=True, exist_ok=True)
    payload = Path("VOICEBOX_COLAB_PKG_B64.txt")
    # Fallback: the next cell-write is inlined below.
    b64 = """__PKG_B64__"""
    raw = base64.b64decode(b64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(dest)
    root = dest
    print("Extracted embedded Voicebox Colab package →", root.resolve())
else:
    print("Using existing package at", root.resolve())

os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"])
print("Core dependencies installed.")
'''
    ),
    md(
        """## 3. Optional engine installs

Run **only the engines you will use**. Each cell is independent.
The Models tab shows `× Not installed` for anything you skip.

Recommended on a free T4:

1. **Kokoro** — always (tiny, preset voices, good smoke test)
2. **Chatterbox Multilingual** — Hindi + 22 other languages, cloning
3. **Chatterbox Turbo** — English tags (`[laugh]`, `[sigh]`, …)
4. **Qwen 0.6B / CustomVoice 0.6B** — if you want instruct or Qwen cloning
"""
    ),
    md("### 3a. Kokoro 82M (recommended first)"),
    code(
        """import subprocess, sys

def pip_install(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

pip_install("kokoro>=0.9.4", "misaki[en,ja,zh]>=0.9.4", "unidic-lite")
pip_install(
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
print("Kokoro install finished.")
"""
    ),
    md(
        """### 3b. Chatterbox Multilingual + Turbo

Voicebox installs `chatterbox-tts` with `--no-deps` because upstream pins
old `numpy` / `torch`. Same recipe here.

Do **not** use a `\\` line continuation with `%pip` — Colab treats the next
line as Python and the install dies. These cells call `python -m pip`
with a real argument list.
"""
    ),
    code(
        """import subprocess, sys

def pip_install(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

# Package itself — skip its broken pins.
pip_install("--no-deps", "chatterbox-tts")

# Sub-deps from Voicebox backend/requirements.txt (do not pin torch/numpy).
pip_install(
    "conformer>=0.3.2",
    "diffusers>=0.29.0",
    "omegaconf",
    "pykakasi",
    "resemble-perth>=1.0.1",
    "s3tokenizer",
    "spacy-pkuseg",
    "pyloudnorm",
)
print("Chatterbox install finished.")
"""
    ),
    md("### 3c. Qwen3-TTS Base + CustomVoice"),
    code(
        """import subprocess, sys

def pip_install(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

pip_install("qwen-tts>=0.0.5")
print("qwen-tts install finished.")
"""
    ),
    md(
        """### 3d. LuxTTS (optional — git deps, English cloning, ~1 GB)

**COLAB LIMITATION:** Voicebox pulls `Zipvoice` and `linacodec` from git
plus a custom `piper-phonemize` index. This cell may fail on some Colab
images; if it does, skip LuxTTS. The studio still runs.
"""
    ),
    code(
        """import subprocess, sys, traceback

def pip_install(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

try:
    pip_install(
        "--find-links",
        "https://k2-fsa.github.io/icefall/piper_phonemize.html",
        "piper-phonemize",
    )
    pip_install("linacodec @ git+https://github.com/ysharma3501/LinaCodec.git")
    pip_install("Zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git")
    print("LuxTTS install finished.")
except Exception:
    traceback.print_exc()
    print("LuxTTS install failed — engine will show as Not installed.")
"""
    ),
    md(
        """### 3e. HumeAI TADA (optional — ~4 GB / ~8 GB)

Voicebox installs `hume-tada` with `--no-deps` and uses a DAC `Snake1d`
shim (`backend/utils/dac_shim.py`, also ported here). TADA 3B wants ~8 GB.
"""
    ),
    code(
        """import subprocess, sys, traceback

def pip_install(*args):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print(" ".join(cmd))
    subprocess.check_call(cmd)

try:
    pip_install("--no-deps", "hume-tada")
    pip_install("torchaudio")
    print("TADA install finished. First load will download codec + weights + ungated Llama tokenizer.")
except Exception:
    traceback.print_exc()
    print("TADA install failed — engine will show as Not installed.")
"""
    ),
    md(
        """## 4. Launch the studio

The Gradio app binds `0.0.0.0` and enables `share=True` so Colab gives you
a public `*.gradio.live` URL. Profiles are written to
`/content/voicebox_colab/profiles/` and never leave this runtime.
"""
    ),
    code(
        """import os, sys
from pathlib import Path

# Make sure the package is importable after the install cells.
for candidate in (Path.cwd(), Path("/content/PARAM")):
    if (candidate / "voicebox_colab").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        os.chdir(candidate)
        break

from voicebox_colab.config import apply_colab_env, set_data_dir
from voicebox_colab.system import probe_system

if Path("/content").exists():
    set_data_dir("/content/voicebox_colab")
apply_colab_env()

info = probe_system()
print(info.gpu_name, f"{info.vram_total_gb:.1f} GB", info.recommendation)

from voicebox_colab.ui.gradio_app import launch

# share=True → public link for Colab. inline=False avoids a cramped iframe.
launch(share=True, server_name="0.0.0.0", server_port=7860, inline=False)
"""
    ),
    md(
        """## How to use

1. **Models** — load one engine. Official VRAM is listed in the table.
2. **Voices** — for cloning engines, create a profile (2–30 s reference).
   Kokoro / CustomVoice use the preset-voice dropdown on Studio instead.
3. **Studio**
   - Pick engine + language (language list is per-engine, from Voicebox).
   - Expression / instruct / tags appear **only** when that engine supports them.
   - Long text mode uses Voicebox's sentence splitter + crossfade.
   - Effects run *after* TTS via Spotify `pedalboard`.
4. **History** — replay / download / delete. Session only.

### Expression mapping (honest)

| Engine | What a preset actually does |
|---|---|
| Qwen CustomVoice | becomes a natural-language `instruct` string |
| Chatterbox Multilingual | becomes Voicebox's `exaggeration` float (0–1) |
| Chatterbox Turbo | ignored — use the tag picker |
| Everything else | control is hidden |

### COLAB LIMITATION (not faked)

Tauri desktop, global dictation hotkey, Stories timeline, MCP agent voice,
Whisper Captures, local personality LLM, MLX, cloud sync. See the About tab.
"""
    ),
]


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "toc_visible": True,
        },
    },
    "cells": cells,
}

pkg_b64 = _package_b64()
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    text = "".join(cell["source"])
    if "__PKG_B64__" not in text:
        continue
    text = text.replace("__PKG_B64__", pkg_b64)
    cell["source"] = [ln + "\n" for ln in text.split("\n")]
    if cell["source"] and cell["source"][-1] == "\n":
        cell["source"].pop()
    break
else:
    raise SystemExit("bootstrap cell missing __PKG_B64__ placeholder")

NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", NB_PATH, "cells=", len(cells), "embedded_pkg_chars=", len(pkg_b64))
