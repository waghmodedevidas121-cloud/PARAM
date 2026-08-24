# PARAM

## Colab notebooks

- **[PARAM AI Upscaler](019ffae2-param/PARAM_Upscaler_Colab.ipynb)** — a Google Colab image and video upscaler designed for the free T4 tier.
- [Voicebox Colab](019ffae2-param/Voicebox_Colab.ipynb) — the existing AI voice studio notebook.

### PARAM AI Upscaler quick start

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order.
4. Open the public Gradio link from the final cell.
5. Upload an image or video, select **2x** or **4x**, and download the result.

The upscaler uses tiled FP16 Real-ESRGAN inference for T4-friendly VRAM usage. **Auto** is the safe default for real photographs; choose **Anime / illustration** for cartoons and line art. GFPGAN face restoration is optional and can change facial details, so compare it with the normal result.

No model checkpoints are committed to the repository. They download into the Colab runtime only when first needed.
