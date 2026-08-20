# PARAM

Google Colab notebooks with browser-based Gradio interfaces for open AI media workflows.

## Notebooks

| Studio | What it does | Runtime | Open |
|---|---|---|---|
| **Open Avatar Studio — Video Twin** | Uses a 2–20 second authorized motion-reference video plus new speech to preserve full-frame body/hand/background motion while generating new facial speech motion. Photo mode is also included. | Colab T4 / L4 / A100 | [![Open Avatar Studio in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waghmodedevidas121-cloud/PARAM/blob/main/Ditto_Avatar_Colab_Gradio.ipynb) |
| **Voicebox Colab** | Multi-engine text-to-speech and voice studio with a Gradio UI. | Colab GPU recommended | [![Open Voicebox in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waghmodedevidas121-cloud/PARAM/blob/main/019ffae2-param/Voicebox_Colab.ipynb) |

## Avatar workflow

1. Open `Ditto_Avatar_Colab_Gradio.ipynb` in Colab.
2. Select a **T4 GPU** runtime and run the cells in order.
3. Open the temporary private `gradio.live` URL.
4. For body motion, upload a 10–15 second reference video with a fixed camera and natural visible gestures.
5. Upload/record the new speech, confirm permission, generate, and download the MP4.

The T4-friendly **Video Twin** mode reuses and mirror-loops recorded full-frame motion while replacing facial speech motion and audio. It is an open Avatar V-lite workflow, not a script-aware generative replacement for HeyGen's proprietary Avatar V. Photo-only mode remains available. The notebook uses a pinned upstream source revision, an isolated Python 3.10 environment, only the required PyTorch checkpoints, input validation, an optional synthetic-media disclosure watermark, and a one-job-at-a-time queue.

> Use only faces, reference videos, and voices you own or have explicit permission to animate. Do not use these notebooks for impersonation, fraud, harassment, or non-consensual synthetic media.
