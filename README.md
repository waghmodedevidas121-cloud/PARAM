# PARAM

Google Colab notebooks with browser-based Gradio interfaces for open AI media workflows.

## Notebooks

| Studio | What it does | Runtime | Open |
|---|---|---|---|
| **Open Avatar Studio — Ditto** | Turns one authorized portrait plus speech audio into an expressive talking-avatar MP4. Uses Ditto's Apache-2.0 PyTorch checkpoint and avoids T4-incompatible prebuilt TensorRT engines. | Colab T4 / L4 / A100 | [![Open Avatar Studio in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waghmodedevidas121-cloud/PARAM/blob/main/Ditto_Avatar_Colab_Gradio.ipynb) |
| **Voicebox Colab** | Multi-engine text-to-speech and voice studio with a Gradio UI. | Colab GPU recommended | [![Open Voicebox in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waghmodedevidas121-cloud/PARAM/blob/main/019ffae2-param/Voicebox_Colab.ipynb) |

## Avatar workflow

1. Open `Ditto_Avatar_Colab_Gradio.ipynb` in Colab.
2. Select a **T4 GPU** runtime and run the cells in order.
3. Open the temporary private `gradio.live` URL.
4. Upload a clear portrait and upload or record speech.
5. Confirm permission, generate, and download the MP4.

The notebook uses a pinned upstream source revision, an isolated Python 3.10 environment, only the required PyTorch checkpoints, input validation, an optional synthetic-media disclosure watermark, and a one-job-at-a-time queue.

> Use only faces and voices you own or have explicit permission to animate. Do not use these notebooks for impersonation, fraud, harassment, or non-consensual synthetic media.
