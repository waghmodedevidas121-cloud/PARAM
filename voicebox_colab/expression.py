"""
Expression presets for the Colab UI.

Voicebox does not ship named emotion classes. Expression is engine-native:

  * Qwen CustomVoice  — natural-language `instruct`  (supports_instruct=True)
  * Chatterbox MTL    — `exaggeration` float 0–1     (PROJECT_STATUS.md)
  * Chatterbox Turbo  — inline paralinguistic tags   (ParalinguisticInput.tsx)
  * All other engines — no expression path

Presets therefore *translate* into whatever the selected engine actually
honours. They are never presented as native emotion labels on engines
that do not have them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .capabilities import get_capabilities


@dataclass(frozen=True)
class ExpressionPreset:
    name: str
    instruct: str
    exaggeration: float
    description: str


# Delivery copy is written as Qwen CustomVoice instruct strings
# (the only engine that consumes them). Exaggeration is the Chatterbox
# Multilingual parameter Voicebox already passes through.
EXPRESSION_PRESETS: dict[str, ExpressionPreset] = {
    "Neutral": ExpressionPreset(
        "Neutral",
        "Speak naturally, with a neutral, even tone.",
        0.35,
        "Even, unmarked delivery.",
    ),
    "Calm": ExpressionPreset(
        "Calm",
        "Speak slowly and calmly.",
        0.25,
        "Low energy, unhurried.",
    ),
    "Conversational": ExpressionPreset(
        "Conversational",
        "Speak naturally, with a confident but conversational tone.",
        0.45,
        "Voicebox README example instruct.",
    ),
    "Happy": ExpressionPreset(
        "Happy",
        "Speak with a happy, warm, and upbeat tone.",
        0.65,
        "Bright, smiling delivery.",
    ),
    "Sad": ExpressionPreset(
        "Sad",
        "Speak softly with emotional warmth and a sad tone.",
        0.30,
        "Quieter, heavier cadence.",
    ),
    "Angry": ExpressionPreset(
        "Angry",
        "Speak in an angry tone, with strong intensity.",
        0.85,
        "High exaggeration on Chatterbox.",
    ),
    "Dramatic": ExpressionPreset(
        "Dramatic",
        "Speak with dramatic intensity.",
        0.75,
        "Staged, theatrical.",
    ),
    "Energetic": ExpressionPreset(
        "Energetic",
        "Speak with high energy and a driving rhythm.",
        0.70,
        "Fast, punched delivery.",
    ),
    "Excited": ExpressionPreset(
        "Excited",
        "Speak like an excited storyteller.",
        0.85,
        "Voicebox README example instruct.",
    ),
    "Serious": ExpressionPreset(
        "Serious",
        "Speak seriously and formally, with strong emphasis on important words.",
        0.40,
        "Measured, formal.",
    ),
    "Cinematic": ExpressionPreset(
        "Cinematic",
        "Speak with cinematic gravitas and measured pacing.",
        0.70,
        "Trailer / narration weight.",
    ),
    "Storytelling": ExpressionPreset(
        "Storytelling",
        "Speak like a warm narrator telling a story.",
        0.55,
        "Narrative cadence.",
    ),
}

PRESET_NAMES = list(EXPRESSION_PRESETS.keys())

# Shown in the CustomVoice delivery dropdown (Voicebox README examples).
DELIVERY_PRESETS: list[str] = [
    "Speak naturally, with a confident but conversational tone.",
    "Speak slowly and calmly.",
    "Speak with dramatic intensity.",
    "Speak like an excited storyteller.",
    "Speak softly with emotional warmth.",
    "Speak with strong emphasis on important words.",
]


def resolve_expression(
    engine: str,
    preset_name: Optional[str],
    custom_instruct: Optional[str] = None,
) -> dict:
    """
    Return the engine-native knobs for this preset.

    Keys:
      instruct          — only set when the engine honours instruct
      exaggeration      — only set when the engine honours exaggeration
      applied           — what actually happened, for the UI status line
      ignored_reason    — set when the preset cannot be applied
    """
    caps = get_capabilities(engine)
    preset = EXPRESSION_PRESETS.get(preset_name or "")

    instruct = (custom_instruct or "").strip() or None
    if instruct and not caps.style_instructions:
        instruct = None

    if caps.style_instructions:
        if not instruct and preset:
            instruct = preset.instruct
        return {
            "instruct": instruct,
            "exaggeration": None,
            "applied": "instruct" if instruct else "none",
            "ignored_reason": None,
        }

    if caps.exaggeration:
        exaggeration = preset.exaggeration if preset else 0.5
        return {
            "instruct": None,
            "exaggeration": exaggeration,
            "applied": f"exaggeration={exaggeration:.2f}",
            "ignored_reason": None,
        }

    if caps.paralinguistic_tags:
        return {
            "instruct": None,
            "exaggeration": None,
            "applied": "tags",
            "ignored_reason": (
                "Chatterbox Turbo has no emotion classes. Use the tag picker "
                "([laugh], [sigh], …) — expression presets are not native here."
            ),
        }

    return {
        "instruct": None,
        "exaggeration": None,
        "applied": "none",
        "ignored_reason": (
            f"{caps.display_name} has no expression path in Voicebox. "
            "Presets are hidden for this engine."
        ),
    }


def engine_has_expression_ui(engine: str) -> bool:
    caps = get_capabilities(engine)
    return caps.style_instructions or caps.exaggeration or caps.paralinguistic_tags
