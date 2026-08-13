# Voicebox Colab

A **Google Colab + Gradio** adaptation of [Jamie Pine Voicebox](https://github.com/jamiepine/voicebox) — the open-source, local-first AI voice studio.

This is **not** a standalone Chatterbox demo. The Voicebox `main` branch is the architectural source of truth for engines, model configs, voice profiles, chunking, effects, and capabilities. The Tauri/React desktop frontend is replaced by a Gradio studio that runs on Colab CUDA.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waghmodedevidas121-cloud/PARAM/blob/arena%2F019ffae2-param/Voicebox_Colab.ipynb)

**Open `Voicebox_Colab.ipynb` in Colab** (GPU runtime). The notebook is self-contained — it does **not** clone GitHub — then installs the engines you opt into and launches the Gradio studio.

> This repository is private and the working branch is `arena/019ffae2-param`.
> GitHub’s contents API must be called with `ref=arena/019ffae2-param` (URL-encoded `arena%2F019ffae2-param`).
> A request for `contents/019ffae2-param?ref=arena` is a bad split of the branch name and will 404.

---

## What was kept from Voicebox

| Voicebox (`main`) | Colab adaptation |
|---|---|
| 7 TTS engines + `ModelConfig` registry | `voicebox_colab/backends/` |
| `TTSBackend` protocol (load / prompt / generate / unload) | Same methods, sync for Gradio |
| Voice profiles (`cloned` / `preset`) | JSON + WAV under `/content/voicebox_colab/profiles/` |
| Sentence-boundary chunking + crossfade | `voicebox_colab/chunked_tts.py` |
| Pedalboard effects + 4 built-in presets | `voicebox_colab/effects.py` |
| Model manager (load / unload / VRAM) | one model at a time |
| Honest capabilities | UI hides unsupported controls |

## Engines (current Voicebox `main`)

| Engine | HF repo | Clone | Languages | Instruct | Tags | VRAM |
|---|---|---|---|---|---|---|
| Qwen3-TTS 1.7B / 0.6B | `Qwen/Qwen3-TTS-12Hz-*-Base` | yes | 10 | **no** (Base drops it) | no | ~6 / ~2 GB |
| Qwen CustomVoice 1.7B / 0.6B | `Qwen/Qwen3-TTS-12Hz-*-CustomVoice` | 9 presets | 10 | **yes** | no | ~6 / ~2 GB |
| LuxTTS | `YatharthS/LuxTTS` | yes | English | no | no | ~1 GB |
| Chatterbox Multilingual | `ResembleAI/chatterbox` | yes | 23 | no | no | ~3 GB |
| Chatterbox Turbo | `ResembleAI/chatterbox-turbo` | yes | English | no | **yes** | ~1.5 GB |
| TADA 1B / 3B-ML | `HumeAI/tada-*` | yes | en / 10 | no | no | ~4 / ~8 GB |
| Kokoro 82M | `hexgrad/Kokoro-82M` | 50 presets | 8 | no | no | ~0.15 GB |

VRAM numbers are from Voicebox `docs/content/docs/developer/model-management.mdx`, not estimates.

**Capability honesty:** Qwen3-TTS Base still *accepts* an `instruct` kwarg in Voicebox's PyTorch backend, but `ModelConfig.supports_instruct=False` and the desktop UI never sends it (“Base model drops instruct silently”). This Colab UI follows that. Delivery instructions appear only for **Qwen CustomVoice**. Paralinguistic tags appear only for **Chatterbox Turbo**. Expression presets translate into `instruct` or Chatterbox `exaggeration` — they are not fake emotion classes.

## Architecture

```
Google Colab
     │
     ├── Voicebox-derived backend logic
     │
     ├── Model Manager          (one model loaded)
     ├── Voice Profile Manager  (/content/voicebox_colab/profiles/)
     ├── TTS Engine Manager     (7 engines, lazy import)
     ├── Audio Processing       (normalize, trim, validate)
     ├── Chunking + Crossfade   (Voicebox splitter)
     └── Gradio UI
              ├── Voice Cloning
              ├── Multilingual TTS
              ├── Expression (engine-native only)
              ├── Advanced Controls
              ├── Audio Effects
              └── Generation History
```

## Quick start (Colab)

1. Runtime → Change runtime type → **GPU** (T4 is enough for any single engine).
2. Run the install cells in `Voicebox_Colab.ipynb`.
3. Run **Launch studio**. Use the public Gradio link if the inline frame is cramped.

Locally:

```bash
pip install -r requirements-colab.txt
# plus at least one engine, e.g.  pip install kokoro 'misaki[en]'
python launch.py --no-share
```

## COLAB LIMITATION

Desktop-only Voicebox features that are **not** faked here:

| Feature | Alternative in this repo |
|---|---|
| Tauri app, global hotkey, auto-paste | Gradio in the browser |
| Stories multi-track timeline | Long-text mode + history |
| MCP server / agent `voicebox.speak` | Generate tab |
| Whisper STT / Captures / dictation | Not ported |
| Local Qwen3 personality LLM | Not ported (VRAM) |
| Voice Design (`designed` profiles) | Not implemented upstream either |
| MLX Apple Silicon | CUDA / CPU only |
| Cloud backup | Local JSON only |
| Generation versions / takes | Session history |

TADA and LuxTTS match Voicebox's backends (including the DAC shim and ungated Llama tokenizer for TADA) but their Python deps are optional. The Models tab marks an engine `× Not installed` instead of pretending it works.

## Layout

```
voicebox_colab/
  capabilities.py     # verified ENGINE_CAPABILITIES
  backends/           # one file per Voicebox engine
  chunked_tts.py      # Voicebox splitter + crossfade
  effects.py          # pedalboard registry + presets
  profiles.py / history.py
  services/           # model manager + generation
  ui/gradio_app.py    # Gradio studio
Voicebox_Colab.ipynb  # Colab entry point
```

## License

MIT, same as Voicebox. This adaptation includes Voicebox copyright and credits [jamiepine/voicebox](https://github.com/jamiepine/voicebox).
