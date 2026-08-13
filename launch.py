#!/usr/bin/env python3
"""Launch the Voicebox Colab Gradio studio (works locally and on Colab)."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Voicebox Colab — Gradio studio")
    parser.add_argument("--host", default=os.environ.get("VOICEBOX_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VOICEBOX_PORT", "7860")))
    parser.add_argument("--share", action="store_true", default=True)
    parser.add_argument("--no-share", dest="share", action="store_false")
    parser.add_argument("--data-dir", default=os.environ.get("VOICEBOX_COLAB_DATA_DIR"))
    args = parser.parse_args()

    from voicebox_colab.config import apply_colab_env, set_data_dir

    if args.data_dir:
        set_data_dir(args.data_dir)
    apply_colab_env()

    from voicebox_colab.ui.gradio_app import launch

    launch(share=args.share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
