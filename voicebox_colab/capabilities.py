"""
Honest engine capability map, verified against Voicebox current main.

Sources:
  - backend/backends/__init__.py          ModelConfig.supports_instruct, languages
  - backend/services/profiles.py          CLONING_ENGINES
  - app/src/lib/hooks/useGenerationForm.ts
        "Only Qwen CustomVoice actually honors the instruct kwarg"
  - app/src/components/Generation/EngineModelSelector.tsx
  - app/src/lib/constants/languages.ts
  - app/src/components/Generation/ParalinguisticInput.tsx
  - README.md  "Emotions & Paralinguistic Tags"
  - docs/content/docs/developer/model-management.mdx  (VRAM table)

Do NOT expose a control unless the selected engine actually uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ISO codes → display names. Union of every engine in Voicebox languages.ts.
ALL_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

# Voicebox backend/backends/__init__.py LANGUAGE_CODE_TO_NAME (Qwen path).
LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}

# Per-engine language lists copied from Voicebox languages.ts / ModelConfig.
ENGINE_LANGUAGES: dict[str, list[str]] = {
    "qwen": ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
    "qwen_custom_voice": ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
    "luxtts": ["en"],
    "chatterbox": [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
        "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    ],
    "chatterbox_turbo": ["en"],
    "tada": ["en", "ar", "zh", "de", "es", "fr", "it", "ja", "pl", "pt"],
    "tada_1b": ["en"],
    "tada_3b": ["en", "ar", "zh", "de", "es", "fr", "it", "ja", "pl", "pt"],
    "kokoro": ["en", "es", "fr", "hi", "it", "pt", "ja", "zh"],
}

# profiles.py CLONING_ENGINES
CLONING_ENGINES = frozenset({"qwen", "luxtts", "chatterbox", "chatterbox_turbo", "tada"})
PRESET_ENGINES = frozenset({"qwen_custom_voice", "kokoro"})

# Chatterbox Turbo tags from ParalinguisticInput.tsx — these are the only
# tags Voicebox inserts. Other engines read them as literal text.
PARALINGUISTIC_TAGS: list[dict[str, str]] = [
    {"tag": "[laugh]", "label": "laugh", "emoji": "😂"},
    {"tag": "[chuckle]", "label": "chuckle", "emoji": "😏"},
    {"tag": "[gasp]", "label": "gasp", "emoji": "😮"},
    {"tag": "[cough]", "label": "cough", "emoji": "😷"},
    {"tag": "[sigh]", "label": "sigh", "emoji": "😔"},
    {"tag": "[groan]", "label": "groan", "emoji": "😩"},
    {"tag": "[sniff]", "label": "sniff", "emoji": "👃"},
    {"tag": "[shush]", "label": "shush", "emoji": "🤫"},
    {"tag": "[clear throat]", "label": "clear throat", "emoji": "🙊"},
]


@dataclass(frozen=True)
class EngineCapabilities:
    """Capability flags for one Voicebox engine. Values verified from source."""

    engine: str
    display_name: str
    cloning: bool
    preset_voices: bool
    multilingual: bool
    style_instructions: bool
    paralinguistic_tags: bool
    exaggeration: bool  # Chatterbox Multilingual float 0–1 (PROJECT_STATUS.md)
    languages: list[str] = field(default_factory=list)
    notes: str = ""


# Verified from Voicebox current main. Do not invent flags.
ENGINE_CAPABILITIES: dict[str, EngineCapabilities] = {
    "qwen": EngineCapabilities(
        engine="qwen",
        display_name="Qwen3-TTS",
        cloning=True,
        preset_voices=False,
        multilingual=True,
        # ModelConfig.supports_instruct=False — "Base model drops instruct silently"
        # useGenerationForm.ts: "Only Qwen CustomVoice actually honors the instruct kwarg"
        style_instructions=False,
        paralinguistic_tags=False,
        exaggeration=False,
        languages=ENGINE_LANGUAGES["qwen"],
        notes="Base checkpoint accepts an instruct kwarg but drops it. Do not expose a delivery-instruction control.",
    ),
    "qwen_custom_voice": EngineCapabilities(
        engine="qwen_custom_voice",
        display_name="Qwen CustomVoice",
        cloning=False,
        preset_voices=True,
        multilingual=True,
        style_instructions=True,  # ModelConfig.supports_instruct=True
        paralinguistic_tags=False,
        exaggeration=False,
        languages=ENGINE_LANGUAGES["qwen_custom_voice"],
        notes="9 preset speakers. Natural-language instruct controls tone / emotion / pace.",
    ),
    "luxtts": EngineCapabilities(
        engine="luxtts",
        display_name="LuxTTS",
        cloning=True,
        preset_voices=False,
        multilingual=False,
        style_instructions=False,
        paralinguistic_tags=False,
        exaggeration=False,
        languages=ENGINE_LANGUAGES["luxtts"],
        notes="English-only zero-shot cloning. ~1 GB VRAM, 48 kHz.",
    ),
    "chatterbox": EngineCapabilities(
        engine="chatterbox",
        display_name="Chatterbox Multilingual",
        cloning=True,
        preset_voices=False,
        multilingual=True,
        style_instructions=False,
        paralinguistic_tags=False,
        exaggeration=True,  # PROJECT_STATUS.md: "Partial — exaggeration float (0-1)"
        languages=ENGINE_LANGUAGES["chatterbox"],
        notes="23 languages. Expression presets map to the exaggeration parameter. Tags are read literally.",
    ),
    "chatterbox_turbo": EngineCapabilities(
        engine="chatterbox_turbo",
        display_name="Chatterbox Turbo",
        cloning=True,
        preset_voices=False,
        multilingual=False,
        style_instructions=False,
        paralinguistic_tags=True,
        exaggeration=False,
        languages=ENGINE_LANGUAGES["chatterbox_turbo"],
        notes="English only. Paralinguistic tags are the native expression system.",
    ),
    "tada": EngineCapabilities(
        engine="tada",
        display_name="TADA",
        cloning=True,
        preset_voices=False,
        multilingual=True,  # 3B only; 1B is English. Variant-checked at runtime.
        style_instructions=False,
        paralinguistic_tags=False,
        exaggeration=False,
        languages=ENGINE_LANGUAGES["tada"],
        notes="1B is English-only; 3B-ML covers 10 languages. No instruct / tags.",
    ),
    "kokoro": EngineCapabilities(
        engine="kokoro",
        display_name="Kokoro 82M",
        cloning=False,
        preset_voices=True,
        multilingual=True,
        style_instructions=False,
        paralinguistic_tags=False,
        exaggeration=False,
        languages=ENGINE_LANGUAGES["kokoro"],
        notes="50 curated preset voices. No zero-shot cloning, no instruct, no tags.",
    ),
}

# Official VRAM from docs/content/docs/developer/model-management.mdx
MODEL_VRAM_GB: dict[str, float] = {
    "qwen-tts-0.6B": 2.0,
    "qwen-tts-1.7B": 6.0,
    "qwen-custom-voice-0.6B": 2.0,
    "qwen-custom-voice-1.7B": 6.0,
    "luxtts": 1.0,
    "chatterbox-tts": 3.0,
    "chatterbox-turbo": 1.5,
    "tada-1b": 4.0,
    "tada-3b-ml": 8.0,
    "kokoro": 0.15,
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "qwen-tts-1.7B": (
        "High-quality multilingual TTS by Alibaba. Supports 10 languages with "
        "natural prosody and voice cloning from short reference audio."
    ),
    "qwen-tts-0.6B": (
        "Lightweight version of Qwen TTS. Same language support with faster "
        "inference, ideal for lower-end hardware."
    ),
    "qwen-custom-voice-1.7B": (
        "Qwen3-TTS CustomVoice 1.7B. 9 premium preset voices with instruct-based "
        "style control for tone, emotion, and prosody. Supports 10 languages."
    ),
    "qwen-custom-voice-0.6B": (
        "Qwen3-TTS CustomVoice 0.6B. Same 9 preset voices and instruct control. "
        "Faster inference for lower-end hardware."
    ),
    "luxtts": (
        "Lightweight ZipVoice-based TTS for high-quality English voice cloning "
        "and 48 kHz speech at speeds exceeding 150x realtime. ~1 GB VRAM."
    ),
    "chatterbox-tts": (
        "Production-grade open-source TTS by Resemble AI. 23 languages with "
        "zero-shot cloning and emotion exaggeration control."
    ),
    "chatterbox-turbo": (
        "Streamlined 350M English TTS by Resemble AI. Paralinguistic tags "
        "([laugh], [sigh], [gasp], …). Lower compute than the multilingual model."
    ),
    "tada-1b": (
        "HumeAI TADA 1B — English speech-language model. 700s+ coherent audio "
        "with text-acoustic dual alignment. ~4 GB VRAM."
    ),
    "tada-3b-ml": (
        "HumeAI TADA 3B Multilingual — 10 languages, high-fidelity cloning via "
        "text-acoustic dual alignment. ~8 GB VRAM."
    ),
    "kokoro": (
        "Kokoro 82M by hexgrad. Tiny 82M-parameter TTS that runs at CPU realtime. "
        "8 languages, 50 pre-built voices. Apache 2.0."
    ),
}


def get_capabilities(engine: str) -> EngineCapabilities:
    if engine not in ENGINE_CAPABILITIES:
        raise KeyError(f"Unknown engine '{engine}'. Known: {list(ENGINE_CAPABILITIES)}")
    return ENGINE_CAPABILITIES[engine]


def languages_for(engine: str, model_size: str | None = None) -> list[str]:
    """TADA 1B is English-only; every other variant uses ENGINE_LANGUAGES."""
    if engine == "tada" and model_size == "1B":
        return ["en"]
    return list(ENGINE_LANGUAGES.get(engine, ["en"]))


def capabilities_dict(engine: str) -> dict[str, Any]:
    caps = get_capabilities(engine)
    return {
        "cloning": caps.cloning,
        "preset_voices": caps.preset_voices,
        "multilingual": caps.multilingual,
        "style_instructions": caps.style_instructions,
        "paralinguistic_tags": caps.paralinguistic_tags,
        "exaggeration": caps.exaggeration,
        "languages": list(caps.languages),
        "notes": caps.notes,
    }
