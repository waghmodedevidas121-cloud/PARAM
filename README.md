# PARAM

Google Colab notebooks with browser-based Gradio interfaces for open AI media workflows.

## Notebooks

| Studio | What it does | Runtime | Open |
|---|---|---|---|
| **OmniAvatar 1.3B Studio** | Generatively creates new facial, head, hand, and body motion from a photo, speech, and behavior prompt. Low-VRAM T4 profile; very slow and 16–24 GB system RAM recommended. | Colab T4 / L4 / A100 | [![Open OmniAvatar in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waghmodedevidas121-cloud/PARAM/blob/main/OmniAvatar_1_3B_T4_Colab_Gradio.ipynb) |
| **Open Avatar Studio — Video Twin** | Uses a 2–20 second authorized motion-reference video plus new speech to preserve full-frame body/hand/background motion while generating new facial speech motion. Photo mode is also included. | Colab T4 / L4 / A100 | [![Open Avatar Studio in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waghmodedevidas121-cloud/PARAM/blob/main/Ditto_Avatar_Colab_Gradio.ipynb) |
| **Voicebox Colab** | Multi-engine text-to-speech and voice studio with a Gradio UI. | Colab GPU recommended | [![Open Voicebox in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waghmodedevidas121-cloud/PARAM/blob/main/019ffae2-param/Voicebox_Colab.ipynb) |

## Generative OmniAvatar workflow

1. Open `OmniAvatar_1_3B_T4_Colab_Gradio.ipynb` and select a T4 GPU; use High RAM when available.
2. Run the pinned setup and model-download cells (roughly 18–19 GB of model files).
3. Open the private Gradio link and start with a waist-up image plus 2–4 seconds of speech.
4. Describe the desired gestures/body behavior in the prompt, then generate at 10–20 steps.

OmniAvatar creates new motion rather than replaying a reference video. T4 inference is research-grade and can take tens of minutes; the 1.3B checkpoint is lower quality than OmniAvatar 14B.

## Video Twin workflow

1. Open `Ditto_Avatar_Colab_Gradio.ipynb` in Colab.
2. Select a **T4 GPU** runtime and run the cells in order.
3. Open the temporary private `gradio.live` URL.
4. For body motion, upload a 10–15 second reference video with a fixed camera and natural visible gestures.
5. Upload/record the new speech, confirm permission, generate, and download the MP4.

The T4-friendly **Video Twin** mode reuses and mirror-loops recorded full-frame motion while replacing facial speech motion and audio. It is an open Avatar V-lite workflow, not a script-aware generative replacement for HeyGen's proprietary Avatar V. Photo-only mode remains available. The notebook uses a pinned upstream source revision, an isolated Python 3.10 environment, only the required PyTorch checkpoints, input validation, an optional synthetic-media disclosure watermark, and a one-job-at-a-time queue.

> Use only faces, reference videos, and voices you own or have explicit permission to animate. Do not use these notebooks for impersonation, fraud, harassment, or non-consensual synthetic media.
