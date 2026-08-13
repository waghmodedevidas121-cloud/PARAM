"""
Premium Gradio UI that replaces Voicebox's Tauri/React frontend.

Controls are shown or hidden from ENGINE_CAPABILITIES — never faked.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from .. import config, history, profiles
from ..backends import (
    ENGINE_DESCRIPTIONS,
    ENGINE_OPTIONS,
    get_model_config,
    parse_engine_selection,
    probe_import,
)
from ..backends.kokoro import KOKORO_DEFAULT_VOICE, voice_choices as kokoro_voices
from ..backends.qwen_custom_voice import (
    QWEN_CV_DEFAULT_SPEAKER,
    voice_choices as qwen_cv_voices,
)
from ..capabilities import (
    ALL_LANGUAGES,
    CLONING_ENGINES,
    MODEL_DESCRIPTIONS,
    PARALINGUISTIC_TAGS,
    PRESET_ENGINES,
    get_capabilities,
    languages_for,
)
from ..effects import BUILTIN_PRESETS, build_effects_chain_from_ui
from ..expression import (
    DELIVERY_PRESETS,
    PRESET_NAMES,
    engine_has_expression_ui,
    resolve_expression,
)
from ..services.generation import record_history, run_generation
from ..services.model_manager import get_model_manager
from ..system import empty_cuda_cache, format_status_markdown, probe_system

logger = logging.getLogger("voicebox_colab.ui")

CSS_PATH = Path(__file__).with_name("theme.css")
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""

ENGINE_CHOICES = [(opt["label"], opt["value"]) for opt in ENGINE_OPTIONS]
DEFAULT_ENGINE_VALUE = "kokoro"


def _lang_choices(engine: str, model_size: Optional[str] = None) -> list[tuple[str, str]]:
    codes = languages_for(engine, model_size)
    return [(f"{ALL_LANGUAGES.get(c, c)} ({c})", c) for c in codes]


def _profile_dropdown_update():
    choices = profiles.profile_choices()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def _history_dropdown_update():
    choices = history.history_labels()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def _models_markdown() -> str:
    rows = get_model_manager().list_status()
    lines = [
        "### MODEL MANAGER",
        "",
        "Only the selected model is loaded. Official VRAM figures are from "
        "Voicebox `docs/content/docs/developer/model-management.mdx`.",
        "",
        "| Model | State | Disk | VRAM | HF repo |",
        "|---|---|---|---|---|",
    ]
    mark = {
        "loaded": "● Loaded",
        "downloaded": "◐ Downloaded",
        "available": "○ Available",
        "unavailable": "× Not installed",
    }
    for row in rows:
        rec = " · recommended" if row["recommended"] and row["state"] != "unavailable" else ""
        lines.append(
            f"| {row['display_name']} | {mark.get(row['state'], row['state'])}{rec} "
            f"| {row['size_mb']} MB | ~{row['vram_gb']} GB | `{row['hf_repo_id']}` |"
        )
    missing = [r for r in rows if not r["available"]]
    if missing:
        lines.append("")
        lines.append("**Not installed in this runtime**")
        for row in missing:
            lines.append(f"- {row['display_name']}: `{row['import_reason']}`")
    return "\n".join(lines)


def _system_md() -> str:
    mgr = get_model_manager()
    current = mgr.loaded_name or "—"
    cfg = get_model_config(current) if current != "—" else None
    label = cfg.display_name if cfg else current
    return format_status_markdown(current_model=label, status=mgr.status)


def _on_engine_change(engine_value: str, language: str):
    engine, model_size, model_name = parse_engine_selection(engine_value)
    caps = get_capabilities(engine)
    langs = _lang_choices(engine, model_size)
    lang_codes = [c for _, c in langs]
    new_lang = language if language in lang_codes else (lang_codes[0] if lang_codes else "en")

    cloning = engine in CLONING_ENGINES
    preset = engine in PRESET_ENGINES

    if engine == "kokoro":
        preset_choices = kokoro_voices(new_lang)
        preset_value = preset_choices[0][1] if preset_choices else KOKORO_DEFAULT_VOICE
    elif engine == "qwen_custom_voice":
        preset_choices = qwen_cv_voices()
        preset_value = QWEN_CV_DEFAULT_SPEAKER
    else:
        preset_choices = []
        preset_value = None

    show_expr = caps.style_instructions or caps.exaggeration
    show_tags = caps.paralinguistic_tags
    show_instruct = caps.style_instructions
    show_exag_note = caps.exaggeration
    show_tag_warn = show_tags

    desc = ENGINE_DESCRIPTIONS.get(engine, "")
    extra = MODEL_DESCRIPTIONS.get(model_name, caps.notes)
    info = (
        f"**{caps.display_name}** — {desc}\n\n{extra}\n\n"
        f"Cloning: {'yes' if caps.cloning else 'no'} · "
        f"Preset voices: {'yes' if caps.preset_voices else 'no'} · "
        f"Instruct: {'yes' if caps.style_instructions else 'no'} · "
        f"Tags: {'yes' if caps.paralinguistic_tags else 'no'} · "
        f"Exaggeration: {'yes' if caps.exaggeration else 'no'}"
    )

    return (
        gr.update(choices=langs, value=new_lang),
        gr.update(visible=cloning),
        gr.update(visible=preset, choices=preset_choices, value=preset_value),
        gr.update(visible=show_expr),
        gr.update(visible=show_instruct),
        gr.update(visible=show_instruct),
        gr.update(visible=show_tags),
        gr.update(visible=show_exag_note),
        gr.update(visible=show_tag_warn),
        info,
        gr.update(value=f"Selected `{model_name}` · {caps.notes}"),
    )


def _refresh_all():
    return _system_md(), _models_markdown(), _profile_dropdown_update(), _history_dropdown_update()


def _load_model(engine_value: str, progress=gr.Progress(track_tqdm=True)):
    engine, model_size, model_name = parse_engine_selection(engine_value)

    def _p(msg):
        progress(0, desc=msg)

    msg = get_model_manager().load(model_name, progress=_p)
    return msg, _system_md(), _models_markdown()


def _unload_model(engine_value: str):
    try:
        _, _, model_name = parse_engine_selection(engine_value)
    except Exception:
        model_name = get_model_manager().loaded_name
    msg = get_model_manager().unload(model_name)
    return msg, _system_md(), _models_markdown()


def _clear_gpu():
    get_model_manager().unload_all()
    msg = empty_cuda_cache()
    return msg, _system_md(), _models_markdown()


def _create_profile(name, audio, language, engine_value, reference_text):
    engine, _, _ = parse_engine_selection(engine_value)
    if engine not in CLONING_ENGINES:
        raise gr.Error(
            f"{engine} is a preset-voice engine. Pick a speaker on the Generate tab "
            "instead of uploading reference audio."
        )
    if audio is None:
        raise gr.Error("Upload a 2–30 second reference WAV/MP3/FLAC.")
    path = audio if isinstance(audio, str) else getattr(audio, "name", None)
    if not path:
        raise gr.Error("Could not read the uploaded audio file.")
    record = profiles.create_cloned_profile(
        name=name,
        audio_path=path,
        language=language or "en",
        engine=engine,
        reference_text=reference_text or "",
    )
    return (
        f"Created cloned profile **{record['name']}** ({record['id'][:8]}…)",
        _profile_dropdown_update(),
        None,
        "",
        "",
    )


def _delete_profile(profile_id):
    if not profile_id:
        raise gr.Error("Select a profile to delete.")
    rec = profiles.get_profile(profile_id)
    if not rec:
        raise gr.Error("Profile not found.")
    profiles.delete_profile(profile_id)
    return f"Deleted **{rec['name']}**.", _profile_dropdown_update()


def _preview_profile(profile_id):
    if not profile_id:
        return None
    rec = profiles.get_profile(profile_id)
    if not rec:
        return None
    return rec.get("reference_audio")


def _resolve_profile(engine_value, profile_id, preset_voice_id, language):
    engine, _, _ = parse_engine_selection(engine_value)
    if engine in PRESET_ENGINES:
        if not preset_voice_id:
            raise gr.Error("Select a preset voice.")
        label = preset_voice_id
        name = f"{engine}:{preset_voice_id}"
        return profiles.create_preset_profile(
            name=name, engine=engine, preset_voice_id=preset_voice_id, language=language or "en"
        ), label
    if not profile_id:
        raise gr.Error("Select or create a cloned voice profile first.")
    rec = profiles.get_profile(profile_id)
    if not rec:
        raise gr.Error("Voice profile not found.")
    return rec, rec.get("name") or "Voice"


def _insert_tag(text, tag):
    text = text or ""
    spacer = "" if (not text or text.endswith(" ") or text.endswith("\n")) else " "
    return text + spacer + tag + " "


def _generate(
    engine_value,
    language,
    profile_id,
    preset_voice_id,
    text,
    expression,
    delivery_preset,
    custom_instruct,
    long_text,
    chunk_size,
    crossfade_ms,
    normalize,
    seed,
    effects_on,
    effects_preset,
    pitch,
    reverb_room,
    reverb_wet,
    delay_s,
    delay_mix,
    chorus_mix,
    compressor_on,
    gain_db,
    hipass,
    lopass,
    progress=gr.Progress(track_tqdm=True),
):
    engine, model_size, model_name = parse_engine_selection(engine_value)
    caps = get_capabilities(engine)
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter text to speak.")

    if caps.paralinguistic_tags is False and any(
        t["tag"] in text for t in PARALINGUISTIC_TAGS
    ):
        gr.Warning(
            f"{caps.display_name} will read paralinguistic tags as literal words. "
            "Switch to Chatterbox Turbo for [laugh] / [sigh] / [gasp]."
        )

    profile, profile_name = _resolve_profile(engine_value, profile_id, preset_voice_id, language)

    instruct_src = (custom_instruct or "").strip()
    if not instruct_src and delivery_preset and delivery_preset != "(none)":
        instruct_src = delivery_preset
    resolved = resolve_expression(engine, expression if expression != "(none)" else None, instruct_src)

    max_chars = int(chunk_size) if long_text else 50000
    fade = int(crossfade_ms) if long_text else 50
    seed_val = int(seed) if seed not in (None, "") else None

    chain = build_effects_chain_from_ui(
        enabled=bool(effects_on),
        preset=effects_preset or "none",
        pitch_semitones=float(pitch or 0),
        reverb_room=float(reverb_room or 0.5),
        reverb_wet=float(reverb_wet or 0),
        delay_seconds=float(delay_s or 0.3),
        delay_mix=float(delay_mix or 0),
        chorus_mix=float(chorus_mix or 0),
        compressor_on=bool(compressor_on),
        gain_db=float(gain_db or 0),
        highpass_hz=float(hipass or 20),
        lowpass_hz=float(lopass or 20000),
    )

    status_bits = [f"Engine {caps.display_name}", f"voice {profile_name}"]
    if resolved["applied"] not in (None, "none"):
        status_bits.append(resolved["applied"])
    if resolved.get("ignored_reason"):
        status_bits.append(resolved["ignored_reason"])

    def _p(msg):
        progress(0, desc=msg)

    try:
        result = run_generation(
            text=text,
            model_name=model_name,
            profile=profile,
            language=language or "en",
            seed=seed_val,
            instruct=resolved.get("instruct"),
            exaggeration=resolved.get("exaggeration"),
            max_chunk_chars=max_chars,
            crossfade_ms=fade,
            normalize=bool(normalize),
            effects_chain=chain,
            progress=_p,
        )
    except Exception as exc:
        logger.exception("Generation failed")
        raise gr.Error(str(exc)) from exc

    expr_label = expression if expression and expression != "(none)" else "—"
    record_history(result, profile_name=profile_name, expression=expr_label, text=text)
    summary = (
        f"Generated **{result['duration']:.2f}s** · {profile_name} · "
        f"{language} · {caps.display_name} · {expr_label}\n\n"
        + " · ".join(status_bits)
    )
    return result["audio_path"], summary, _history_dropdown_update(), _system_md()


def _replay_history(entry_id):
    if not entry_id:
        return None, "Select a history item."
    item = history.get_entry(entry_id)
    if not item:
        return None, "History item not found."
    return item.get("audio_path"), (
        f"**{item.get('profile_name')}** — {item.get('language')} — "
        f"{item.get('expression')}\n\n{item.get('text')}"
    )


def _delete_history(entry_id):
    if not entry_id:
        raise gr.Error("Select a history item.")
    history.delete_entry(entry_id)
    return None, "Deleted.", _history_dropdown_update()


def build_app() -> gr.Blocks:
    config.apply_colab_env()
    info = probe_system()
    theme = _studio_theme()

    # Gradio 4/5 accept theme+css on Blocks; Gradio 6 moved them to launch().
    blocks_kw: dict[str, Any] = {"title": "Voicebox Colab"}
    blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    if "fill_height" in blocks_params:
        blocks_kw["fill_height"] = True
    if "theme" in blocks_params:
        blocks_kw["theme"] = theme
    if "css" in blocks_params:
        blocks_kw["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kw) as demo:
        with gr.Row(elem_id="vb-header"):
            with gr.Column(scale=3, elem_id="vb-title"):
                gr.Markdown(
                    """
# Voicebox Colab
The open-source AI voice studio — Google Colab + Gradio adaptation of
[jamiepine/voicebox](https://github.com/jamiepine/voicebox).
Seven TTS engines, honest capabilities, Voicebox chunking + pedalboard effects.
                    """
                )
            with gr.Column(scale=2):
                system_md = gr.Markdown(_system_md(), elem_id="vb-status")

        with gr.Tabs():
            with gr.Tab("Studio"):
                with gr.Row():
                    with gr.Column(scale=2):
                        engine = gr.Dropdown(
                            label="Engine / model",
                            choices=ENGINE_CHOICES,
                            value=DEFAULT_ENGINE_VALUE,
                            info="Same selector as Voicebox EngineModelSelector.tsx",
                        )
                        engine_info = gr.Markdown(
                            "Select an engine. Controls below appear only when that engine supports them."
                        )
                        language = gr.Dropdown(
                            label="Language",
                            choices=_lang_choices("kokoro"),
                            value="en",
                        )
                        profile = gr.Dropdown(
                            label="Cloned voice profile",
                            choices=profiles.profile_choices(),
                            value=None,
                            allow_custom_value=False,
                            visible=False,
                        )
                        preset_voice = gr.Dropdown(
                            label="Preset voice",
                            choices=kokoro_voices("en"),
                            value=KOKORO_DEFAULT_VOICE,
                            visible=True,
                        )
                        text = gr.Textbox(
                            label="Text",
                            lines=7,
                            placeholder='Bro, I didn\'t expect that [gasp] this is insane!',
                            max_lines=24,
                        )
                        with gr.Group(visible=False) as tag_group:
                            gr.Markdown("**Paralinguistic tags** — Chatterbox Turbo only")
                            with gr.Row(elem_classes=["tag-row"]):
                                tag_buttons = []
                                for spec in PARALINGUISTIC_TAGS:
                                    btn = gr.Button(
                                        f"{spec['emoji']} {spec['label']}",
                                        size="sm",
                                    )
                                    tag_buttons.append((btn, spec["tag"]))
                            tag_warn = gr.Markdown(
                                "Tags are inserted into the text. Other engines will speak them as words."
                            )

                        with gr.Group(visible=False) as expr_group:
                            expression = gr.Dropdown(
                                label="Expression preset",
                                choices=["(none)"] + PRESET_NAMES,
                                value="Conversational",
                                info="Translated into instruct (CustomVoice) or exaggeration (Chatterbox).",
                            )
                            exag_note = gr.Markdown(
                                visible=False,
                                value="Chatterbox Multilingual has no emotion classes. "
                                "The preset is mapped to Voicebox's `exaggeration` float (0–1).",
                            )

                        with gr.Group(visible=False) as instruct_group:
                            delivery_preset = gr.Dropdown(
                                label="Delivery instruction preset",
                                choices=["(none)"] + DELIVERY_PRESETS,
                                value="(none)",
                            )
                            custom_instruct = gr.Textbox(
                                label="Custom delivery instruction",
                                lines=2,
                                placeholder="Speak naturally, with a confident but conversational tone.",
                                info="Honoured only by Qwen CustomVoice (supports_instruct=True).",
                            )

                        with gr.Accordion("Long text (Voicebox chunking + crossfade)", open=False):
                            long_text = gr.Checkbox(label="Long Text Mode", value=True)
                            with gr.Row():
                                chunk_size = gr.Slider(
                                    100, 5000, value=800, step=50, label="Chunk size (chars)"
                                )
                                crossfade_ms = gr.Slider(
                                    0, 500, value=50, step=5, label="Crossfade (ms)"
                                )
                            gr.Markdown(
                                "Voicebox splits on sentence boundaries, generates each chunk, "
                                "then crossfades. Abbreviations, CJK punctuation, and `[tags]` "
                                "are never split. Default chunk 800 / crossfade 50 ms."
                            )

                        with gr.Accordion("Audio effects (pedalboard, after TTS)", open=False):
                            effects_on = gr.Checkbox(label="Effects ON", value=False)
                            effects_preset = gr.Dropdown(
                                label="Built-in preset",
                                choices=["custom"] + [p["name"] for p in BUILTIN_PRESETS.values()],
                                value="custom",
                            )
                            with gr.Row():
                                pitch = gr.Slider(-12, 12, 0, step=0.5, label="Pitch (semitones)")
                                gain_db = gr.Slider(-40, 40, 0, step=0.5, label="Gain (dB)")
                            with gr.Row():
                                reverb_room = gr.Slider(0, 1, 0.5, step=0.01, label="Reverb room")
                                reverb_wet = gr.Slider(0, 1, 0.0, step=0.01, label="Reverb wet")
                            with gr.Row():
                                delay_s = gr.Slider(0.01, 2.0, 0.3, step=0.01, label="Delay (s)")
                                delay_mix = gr.Slider(0, 1, 0.0, step=0.01, label="Delay mix")
                            with gr.Row():
                                chorus_mix = gr.Slider(0, 1, 0.0, step=0.01, label="Chorus / flanger mix")
                                compressor_on = gr.Checkbox(label="Compressor", value=False)
                            with gr.Row():
                                hipass = gr.Slider(20, 8000, 20, step=1, label="High-pass (Hz)")
                                lopass = gr.Slider(200, 20000, 20000, step=10, label="Low-pass (Hz)")

                        with gr.Row():
                            normalize = gr.Checkbox(label="Normalize", value=True)
                            seed = gr.Number(label="Seed (optional)", precision=0, value=None)

                        generate_btn = gr.Button("Generate speech", variant="primary", elem_id="generate-btn")
                        engine_note = gr.Markdown("")

                    with gr.Column(scale=1):
                        audio_out = gr.Audio(label="Output", type="filepath", interactive=False)
                        result_md = gr.Markdown("Ready.")
                        refresh_btn = gr.Button("Refresh status", variant="secondary")

            with gr.Tab("Voices"):
                gr.Markdown(
                    """
### Voice profiles
Cloned profiles store a 2–30 s reference WAV under
`/content/voicebox_colab/profiles/` (or `./data/voicebox_colab/profiles/` locally).
Nothing is uploaded to an external server.

Preset engines (Kokoro, Qwen CustomVoice) do **not** clone — pick a speaker
on the Studio tab. That matches Voicebox `voice_type: cloned | preset`.
                    """
                )
                with gr.Row():
                    with gr.Column():
                        vp_name = gr.Textbox(label="Name", placeholder="My Voice")
                        vp_engine = gr.Dropdown(
                            label="Cloning engine (Voicebox CLONING_ENGINES)",
                            choices=[
                                ("Chatterbox Multilingual", "chatterbox"),
                                ("Chatterbox Turbo", "chatterbox_turbo"),
                                ("Qwen3-TTS", "qwen"),
                                ("LuxTTS", "luxtts"),
                                ("TADA", "tada"),
                            ],
                            value="chatterbox",
                            info="Preset engines (Kokoro, CustomVoice) do not take reference audio.",
                        )
                        vp_lang = gr.Dropdown(
                            label="Language tag",
                            choices=[(v, k) for k, v in ALL_LANGUAGES.items()],
                            value="en",
                        )
                        vp_audio = gr.Audio(label="Reference audio (2–30 s)", type="filepath")
                        vp_ref = gr.Textbox(
                            label="Reference transcript (optional)",
                            lines=2,
                            info="Used by Qwen / TADA prompt encoding. Chatterbox ignores it.",
                        )
                        vp_create = gr.Button("Create cloned profile", variant="primary")
                        vp_status = gr.Markdown("")
                    with gr.Column():
                        vp_select = gr.Dropdown(
                            label="Existing profiles",
                            choices=profiles.profile_choices(),
                            value=None,
                        )
                        vp_preview = gr.Audio(label="Preview reference", type="filepath")
                        vp_delete = gr.Button("Delete profile", variant="stop")

            with gr.Tab("Models"):
                models_md = gr.Markdown(_models_markdown())
                with gr.Row():
                    load_btn = gr.Button("Load selected model", variant="primary")
                    unload_btn = gr.Button("Unload selected model")
                    clear_btn = gr.Button("Clear GPU memory", variant="stop")
                model_msg = gr.Markdown(
                    f"Detected **{info.gpu_name}** · {info.vram_total_gb:.1f} GB. {info.recommendation}"
                )
                gr.Markdown(
                    """
**VRAM strategy (Voicebox docs, not guesses)**

| VRAM | Fits |
|---|---|
| ~0.15 GB | Kokoro 82M |
| ~1 GB | LuxTTS |
| ~1.5 GB | Chatterbox Turbo |
| ~2 GB | Qwen 0.6B / CustomVoice 0.6B |
| ~3 GB | Chatterbox Multilingual |
| ~4 GB | TADA 1B |
| ~6 GB | Qwen 1.7B / CustomVoice 1.7B |
| ~8 GB | TADA 3B Multilingual |

Colab free T4 (~15 GB) can hold any *single* engine. Do not load two at once.
                    """
                )

            with gr.Tab("History"):
                gr.Markdown("Session-only. Files live in the Colab data directory and vanish with the runtime.")
                hist_select = gr.Dropdown(
                    label="Recent generations",
                    choices=history.history_labels(),
                    value=None,
                    elem_id="history-list",
                )
                hist_audio = gr.Audio(label="Replay", type="filepath")
                hist_md = gr.Markdown("")
                with gr.Row():
                    hist_play = gr.Button("Replay")
                    hist_del = gr.Button("Delete", variant="stop")

            with gr.Tab("About / limitations"):
                gr.Markdown(
                    """
## This is Voicebox's stack, not a Chatterbox demo

Architecture kept from [jamiepine/voicebox](https://github.com/jamiepine/voicebox) `main`:

- Multi-engine TTS protocol (`qwen`, `qwen_custom_voice`, `luxtts`, `chatterbox`, `chatterbox_turbo`, `tada`, `kokoro`)
- `ModelConfig` registry (HF repos, sizes, languages, `needs_trim`, `supports_instruct`)
- Voice profiles (`cloned` vs `preset`)
- Sentence-boundary chunking + crossfade (`backend/utils/chunked_tts.py`)
- Pedalboard effects + the four built-in presets
- Chatterbox `exaggeration`, CustomVoice `instruct`, Turbo `[tags]`
- Official VRAM table from `docs/content/docs/developer/model-management.mdx`

Frontend replacement: **Gradio**, running on **Google Colab + CUDA**.

### Capability honesty

| Engine | Clone | Multilingual | Instruct | Tags | Exaggeration |
|---|---|---|---|---|---|
| Qwen3-TTS Base | yes | 10 | **no** (Base drops it) | no | no |
| Qwen CustomVoice | no (9 presets) | 10 | **yes** | no | no |
| LuxTTS | yes | English | no | no | no |
| Chatterbox MTL | yes | 23 | no | no | **yes** (0–1) |
| Chatterbox Turbo | yes | English | no | **yes** | no |
| TADA 1B / 3B | yes | 1B en / 3B 10 | no | no | no |
| Kokoro | no (50 presets) | 8 | no | no | no |

Qwen3-TTS Base still *accepts* an instruct kwarg in Voicebox's PyTorch backend,
but `ModelConfig.supports_instruct=False` and the frontend never sends it
("Base model drops instruct silently"). This Colab UI follows that.

### COLAB LIMITATION

These Voicebox features are desktop/Tauri-specific or too heavy for a
notebook session. Closest alternative in parentheses.

| Feature | Status |
|---|---|
| Tauri desktop shell / global hotkey / auto-paste | COLAB LIMITATION |
| Stories multi-track timeline | COLAB LIMITATION (use long-text + history) |
| MCP server / `voicebox.speak` agent binding | COLAB LIMITATION |
| Whisper STT, Captures, dictation refinement | COLAB LIMITATION |
| Local Qwen3 LLM personality rewrite | COLAB LIMITATION (VRAM; not wired) |
| Voice Design (`voice_type=designed`) | Not implemented upstream either |
| MLX Apple Silicon backend | COLAB LIMITATION (CUDA/CPU only) |
| Cloud backup / sync | COLAB LIMITATION (local JSON only) |
| Generation versions / takes | Simplified to session history |
| SQLite profile DB | JSON + WAV under the data dir |

TADA is implemented from Voicebox's `hume_backend.py` (DAC shim + ungated
Llama tokenizer). Install is optional — the Models tab marks it unavailable
if `hume-tada` is missing. Same for LuxTTS (`zipvoice`).
                    """
                )

        # --- wiring ---
        engine_outputs = [
            language,
            profile,
            preset_voice,
            expr_group,
            instruct_group,
            delivery_preset,
            tag_group,
            exag_note,
            tag_warn,
            engine_info,
            engine_note,
        ]

        engine.change(_on_engine_change, [engine, language], engine_outputs)
        language.change(
            lambda ev, lang: _on_engine_change(ev, lang)[2],
            [engine, language],
            [preset_voice],
        )

        for btn, tag in tag_buttons:
            btn.click(_insert_tag, [text, gr.State(tag)], [text])

        generate_btn.click(
            _generate,
            [
                engine, language, profile, preset_voice, text,
                expression, delivery_preset, custom_instruct,
                long_text, chunk_size, crossfade_ms, normalize, seed,
                effects_on, effects_preset, pitch, reverb_room, reverb_wet,
                delay_s, delay_mix, chorus_mix, compressor_on, gain_db, hipass, lopass,
            ],
            [audio_out, result_md, hist_select, system_md],
        )

        refresh_btn.click(_refresh_all, None, [system_md, models_md, profile, hist_select])
        load_btn.click(_load_model, [engine], [model_msg, system_md, models_md])
        unload_btn.click(_unload_model, [engine], [model_msg, system_md, models_md])
        clear_btn.click(_clear_gpu, None, [model_msg, system_md, models_md])

        vp_create.click(
            _create_profile,
            [vp_name, vp_audio, vp_lang, vp_engine, vp_ref],
            [vp_status, profile, vp_audio, vp_name, vp_ref],
        ).then(lambda: _profile_dropdown_update(), None, [vp_select])
        vp_select.change(_preview_profile, [vp_select], [vp_preview])
        vp_delete.click(_delete_profile, [vp_select], [vp_status, vp_select]).then(
            lambda: _profile_dropdown_update(), None, [profile]
        )

        hist_play.click(_replay_history, [hist_select], [hist_audio, hist_md])
        hist_select.change(_replay_history, [hist_select], [hist_audio, hist_md])
        hist_del.click(_delete_history, [hist_select], [hist_audio, hist_md, hist_select])

        demo.load(_on_engine_change, [engine, language], engine_outputs)

    return demo


def _studio_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.zinc,
        font=gr.themes.GoogleFont("IBM Plex Sans"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        body_background_fill="#07080b",
        body_background_fill_dark="#07080b",
        block_background_fill="#10131a",
        block_background_fill_dark="#10131a",
        border_color_primary="#1f2633",
        button_primary_background_fill="#7c9cff",
        button_primary_text_color="#0b1020",
    )


def launch(share: bool = True, server_name: str = "0.0.0.0", server_port: int = 7860, **kwargs):
    config.apply_colab_env()
    demo = build_app()
    launch_kw = dict(share=share, server_name=server_name, server_port=server_port, **kwargs)
    try:
        params = inspect.signature(demo.launch).parameters
    except (TypeError, ValueError):
        params = {}
    if "theme" in params and "theme" not in launch_kw:
        launch_kw["theme"] = _studio_theme()
    if "css" in params and "css" not in launch_kw:
        launch_kw["css"] = CUSTOM_CSS
    if "ssr_mode" in params and "ssr_mode" not in launch_kw:
        # Colab / iframe previews are more reliable without SSR.
        launch_kw["ssr_mode"] = False
    if "allowed_paths" in params and "allowed_paths" not in launch_kw:
        launch_kw["allowed_paths"] = [str(config.get_data_dir())]
    return demo.queue().launch(**launch_kw)
