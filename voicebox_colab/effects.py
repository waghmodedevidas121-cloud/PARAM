"""
Post-processing effects — ported from Voicebox backend/utils/effects.py.

Same eight pedalboard effects, same parameter ranges, same built-in
presets (Robotic, Radio, Echo Chamber, Deep Voice). Applied AFTER TTS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from pedalboard import (
        Chorus,
        Compressor,
        Delay,
        Gain,
        HighpassFilter,
        LowpassFilter,
        Pedalboard,
        PitchShift,
        Reverb,
    )

    _PEDALBOARD_OK = True
except ImportError:  # pragma: no cover
    _PEDALBOARD_OK = False
    Chorus = Compressor = Delay = Gain = None  # type: ignore
    HighpassFilter = LowpassFilter = Pedalboard = PitchShift = Reverb = None  # type: ignore


EFFECT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "chorus": {
        "cls": Chorus,
        "label": "Chorus / Flanger",
        "description": "Modulated delay. Short centre_delay_ms (<10) is flanger; longer is chorus.",
        "params": {
            "rate_hz": {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01},
            "depth": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            "feedback": {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01},
            "centre_delay_ms": {"default": 7.0, "min": 0.5, "max": 50.0, "step": 0.1},
            "mix": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
        },
    },
    "reverb": {
        "cls": Reverb,
        "label": "Reverb",
        "description": "Room reverb.",
        "params": {
            "room_size": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            "damping": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            "wet_level": {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01},
            "dry_level": {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01},
            "width": {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
        },
    },
    "delay": {
        "cls": Delay,
        "label": "Delay",
        "description": "Echo / delay line.",
        "params": {
            "delay_seconds": {"default": 0.3, "min": 0.01, "max": 2.0, "step": 0.01},
            "feedback": {"default": 0.3, "min": 0.0, "max": 0.95, "step": 0.01},
            "mix": {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
        },
    },
    "compressor": {
        "cls": Compressor,
        "label": "Compressor",
        "description": "Dynamic range compression.",
        "params": {
            "threshold_db": {"default": -20.0, "min": -60.0, "max": 0.0, "step": 0.5},
            "ratio": {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1},
            "attack_ms": {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1},
            "release_ms": {"default": 100.0, "min": 10.0, "max": 1000.0, "step": 1.0},
        },
    },
    "gain": {
        "cls": Gain,
        "label": "Gain",
        "description": "Volume adjustment in decibels.",
        "params": {
            "gain_db": {"default": 0.0, "min": -40.0, "max": 40.0, "step": 0.5},
        },
    },
    "highpass": {
        "cls": HighpassFilter,
        "label": "High-Pass Filter",
        "description": "Removes frequencies below the cutoff.",
        "params": {
            "cutoff_frequency_hz": {"default": 80.0, "min": 20.0, "max": 8000.0, "step": 1.0},
        },
    },
    "lowpass": {
        "cls": LowpassFilter,
        "label": "Low-Pass Filter",
        "description": "Removes frequencies above the cutoff.",
        "params": {
            "cutoff_frequency_hz": {"default": 8000.0, "min": 200.0, "max": 20000.0, "step": 1.0},
        },
    },
    "pitch_shift": {
        "cls": PitchShift,
        "label": "Pitch Shift",
        "description": "Shift pitch up or down by semitones.",
        "params": {
            "semitones": {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5},
        },
    },
}


BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {"name": "None", "effects_chain": []},
    "robotic": {
        "name": "Robotic",
        "effects_chain": [
            {
                "type": "chorus",
                "enabled": True,
                "params": {
                    "rate_hz": 0.2,
                    "depth": 1.0,
                    "feedback": 0.35,
                    "centre_delay_ms": 7.0,
                    "mix": 0.5,
                },
            },
        ],
    },
    "radio": {
        "name": "Radio",
        "effects_chain": [
            {"type": "highpass", "enabled": True, "params": {"cutoff_frequency_hz": 300.0}},
            {"type": "lowpass", "enabled": True, "params": {"cutoff_frequency_hz": 3500.0}},
            {
                "type": "compressor",
                "enabled": True,
                "params": {"threshold_db": -15.0, "ratio": 6.0, "attack_ms": 5.0, "release_ms": 50.0},
            },
            {"type": "gain", "enabled": True, "params": {"gain_db": 6.0}},
        ],
    },
    "echo_chamber": {
        "name": "Echo Chamber",
        "effects_chain": [
            {
                "type": "reverb",
                "enabled": True,
                "params": {
                    "room_size": 0.85,
                    "damping": 0.3,
                    "wet_level": 0.45,
                    "dry_level": 0.55,
                    "width": 1.0,
                },
            },
            {
                "type": "delay",
                "enabled": True,
                "params": {"delay_seconds": 0.25, "feedback": 0.3, "mix": 0.2},
            },
        ],
    },
    "deep_voice": {
        "name": "Deep Voice",
        "effects_chain": [
            {"type": "pitch_shift", "enabled": True, "params": {"semitones": -3.0}},
            {"type": "lowpass", "enabled": True, "params": {"cutoff_frequency_hz": 6000.0}},
            {
                "type": "compressor",
                "enabled": True,
                "params": {
                    "threshold_db": -18.0,
                    "ratio": 3.0,
                    "attack_ms": 10.0,
                    "release_ms": 150.0,
                },
            },
        ],
    },
}


def validate_effects_chain(effects_chain: List[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(effects_chain, list):
        return "effects_chain must be a list"
    for i, effect in enumerate(effects_chain):
        if not isinstance(effect, dict):
            return f"Effect at index {i} must be a dict"
        effect_type = effect.get("type")
        if effect_type not in EFFECT_REGISTRY:
            return f"Unknown effect type '{effect_type}' at index {i}"
        params = effect.get("params", {})
        if not isinstance(params, dict):
            return f"Effect '{effect_type}' at index {i}: params must be a dict"
        registry = EFFECT_REGISTRY[effect_type]
        for param_name, value in params.items():
            if param_name not in registry["params"]:
                return f"Effect '{effect_type}' at index {i}: unknown param '{param_name}'"
            pdef = registry["params"][param_name]
            if not isinstance(value, (int, float)):
                return f"Effect '{effect_type}' at index {i}: param '{param_name}' must be a number"
            if value < pdef["min"] or value > pdef["max"]:
                return (
                    f"Effect '{effect_type}' at index {i}: param '{param_name}' "
                    f"must be between {pdef['min']} and {pdef['max']} (got {value})"
                )
    return None


def build_effects_chain_from_ui(
    *,
    enabled: bool,
    preset: str,
    pitch_semitones: float = 0.0,
    reverb_room: float = 0.5,
    reverb_wet: float = 0.33,
    delay_seconds: float = 0.3,
    delay_mix: float = 0.0,
    chorus_mix: float = 0.0,
    compressor_on: bool = False,
    gain_db: float = 0.0,
    highpass_hz: float = 20.0,
    lowpass_hz: float = 20000.0,
) -> List[Dict[str, Any]]:
    """Compose a Voicebox-style effects_chain from Gradio controls."""
    if not enabled:
        return []
    if preset and preset not in ("none", "custom", "None", "Custom"):
        key = preset.lower().replace(" ", "_")
        builtin = BUILTIN_PRESETS.get(key)
        if builtin:
            return list(builtin["effects_chain"])

    chain: List[Dict[str, Any]] = []
    if abs(pitch_semitones) > 0.01:
        chain.append(
            {"type": "pitch_shift", "enabled": True, "params": {"semitones": float(pitch_semitones)}}
        )
    if reverb_wet > 0.01:
        chain.append(
            {
                "type": "reverb",
                "enabled": True,
                "params": {
                    "room_size": float(reverb_room),
                    "damping": 0.5,
                    "wet_level": float(reverb_wet),
                    "dry_level": max(0.0, 1.0 - float(reverb_wet)),
                    "width": 1.0,
                },
            }
        )
    if delay_mix > 0.01:
        chain.append(
            {
                "type": "delay",
                "enabled": True,
                "params": {
                    "delay_seconds": float(delay_seconds),
                    "feedback": 0.3,
                    "mix": float(delay_mix),
                },
            }
        )
    if chorus_mix > 0.01:
        chain.append(
            {
                "type": "chorus",
                "enabled": True,
                "params": {
                    "rate_hz": 1.0,
                    "depth": 0.5,
                    "feedback": 0.0,
                    "centre_delay_ms": 7.0,
                    "mix": float(chorus_mix),
                },
            }
        )
    if compressor_on:
        chain.append(
            {
                "type": "compressor",
                "enabled": True,
                "params": {
                    "threshold_db": -20.0,
                    "ratio": 4.0,
                    "attack_ms": 10.0,
                    "release_ms": 100.0,
                },
            }
        )
    if abs(gain_db) > 0.01:
        chain.append({"type": "gain", "enabled": True, "params": {"gain_db": float(gain_db)}})
    if highpass_hz > 25.0:
        chain.append(
            {
                "type": "highpass",
                "enabled": True,
                "params": {"cutoff_frequency_hz": float(highpass_hz)},
            }
        )
    if lowpass_hz < 19900.0:
        chain.append(
            {
                "type": "lowpass",
                "enabled": True,
                "params": {"cutoff_frequency_hz": float(lowpass_hz)},
            }
        )
    return chain


def apply_effects(
    audio: np.ndarray,
    sample_rate: int,
    effects_chain: List[Dict[str, Any]],
) -> np.ndarray:
    if not effects_chain:
        return audio
    if not _PEDALBOARD_OK:
        raise RuntimeError(
            "pedalboard is not installed. `pip install pedalboard` to enable Voicebox effects."
        )
    error = validate_effects_chain(effects_chain)
    if error:
        raise ValueError(error)

    plugins = []
    for effect in effects_chain:
        if not effect.get("enabled", True):
            continue
        registry = EFFECT_REGISTRY[effect["type"]]
        cls = registry["cls"]
        params = {}
        for pname, pdef in registry["params"].items():
            params[pname] = effect.get("params", {}).get(pname, pdef["default"])
        plugins.append(cls(**params))

    board = Pedalboard(plugins)
    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio
    processed = board(audio_2d.astype(np.float32), sample_rate)
    if audio.ndim == 1:
        return processed[0]
    return processed
