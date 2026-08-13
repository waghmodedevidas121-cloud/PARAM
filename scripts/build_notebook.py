#!/usr/bin/env python3
"""Generate Voicebox_Colab.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "Voicebox_Colab.ipynb"


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

If you opened only the notebook, this clones the repo. If the package is
already next to the notebook (uploaded folder / Drive mount), it uses that.
"""
    ),
    code(
        r"""import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/waghmodedevidas121-cloud/PARAM.git"
BRANCH = "arena/019ffae2-param"
ROOT_CANDIDATES = [
    Path.cwd(),
    Path("/content/PARAM"),
    Path("/content/voicebox-colab"),
]

def find_pkg() -> Path | None:
    for root in ROOT_CANDIDATES:
        if (root / "voicebox_colab" / "ui" / "gradio_app.py").exists():
            return root
    return None

root = find_pkg()
if root is None:
    dest = Path("/content/PARAM")
    print("Package not found next to the notebook — cloning", REPO_URL)
    if dest.exists():
        subprocess.check_call(["git", "-C", str(dest), "pull", "--ff-only"])
    else:
        try:
            subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(dest)])
        except subprocess.CalledProcessError:
            subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(dest)])
    root = dest

os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
print("Using package at", root.resolve())

%pip install -q -r requirements-colab.txt
print("Core dependencies installed.")
"""
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
        """%pip install -q "kokoro>=0.9.4" "misaki[en,ja,zh]>=0.9.4" unidic-lite
%pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
print("Kokoro install finished.")
"""
    ),
    md(
        """### 3b. Chatterbox Multilingual + Turbo

Voicebox installs `chatterbox-tts` with `--no-deps` because the package
pins old `numpy` / `torch`. Same recipe here.
"""
    ),
    code(
        """%pip install -q --no-deps chatterbox-tts
%pip install -q "conformer>=0.3.2" "diffusers>=0.29.0" omegaconf pykakasi \\
    "resemble-perth>=1.0.1" s3tokenizer spacy-pkuseg pyloudnorm
print("Chatterbox install finished.")
"""
    ),
    md("### 3c. Qwen3-TTS Base + CustomVoice"),
    code(
        """%pip install -q "qwen-tts>=0.0.5"
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
        """import traceback
try:
    %pip install -q --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html piper-phonemize
    %pip install -q "linacodec @ git+https://github.com/ysharma3501/LinaCodec.git"
    %pip install -q "Zipvoice @ git+https://github.com/ysharma3501/LuxTTS.git"
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
        """import traceback
try:
    %pip install -q --no-deps hume-tada
    %pip install -q torchaudio
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

NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", NB_PATH, "cells=", len(cells))
