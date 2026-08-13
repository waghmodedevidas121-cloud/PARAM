"""
Voicebox Colab — a Google Colab + Gradio adaptation of Jamie Pine Voicebox.

Architectural source of truth:
    https://github.com/jamiepine/voicebox  (current main)

This package keeps Voicebox's multi-engine TTS stack, model registry,
voice-profile concepts, chunked generation, pedalboard effects, and
honest per-engine capabilities. The Tauri/React desktop frontend is
replaced by Gradio. Desktop-only features are marked COLAB LIMITATION.
"""

from .config import get_data_dir, set_data_dir

__version__ = "0.1.0"
__voicebox_upstream__ = "https://github.com/jamiepine/voicebox"
__all__ = ["get_data_dir", "set_data_dir", "__version__"]
