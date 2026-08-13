"""
Simplified Voicebox-style voice profiles.

Desktop Voicebox stores profiles in SQLite (services/profiles.py).
Colab stores them as JSON + WAV under /content/voicebox_colab/profiles/
and never uploads them anywhere.

Voice types match Voicebox:
  cloned  — reference audio, used by CLONING_ENGINES
  preset  — engine-owned voice id (Kokoro, Qwen CustomVoice)
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config
from .audio import save_audio, validate_and_load_reference_audio
from .capabilities import CLONING_ENGINES, PRESET_ENGINES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _index_path() -> Path:
    return config.get_profiles_dir() / "index.json"


def _load_index() -> list[dict[str, Any]]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_index(items: list[dict[str, Any]]) -> None:
    _index_path().write_text(json.dumps(items, indent=2), encoding="utf-8")


def list_profiles() -> list[dict[str, Any]]:
    return _load_index()


def get_profile(profile_id: str) -> Optional[dict[str, Any]]:
    for item in _load_index():
        if item["id"] == profile_id:
            return item
    return None


def get_profile_by_name(name: str) -> Optional[dict[str, Any]]:
    needle = (name or "").strip().lower()
    for item in _load_index():
        if item.get("name", "").strip().lower() == needle:
            return item
    return None


def create_cloned_profile(
    name: str,
    audio_path: str,
    language: str = "en",
    engine: str = "chatterbox",
    reference_text: str = "",
) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise ValueError("Profile name is required.")
    if get_profile_by_name(name):
        raise ValueError(f"A profile named '{name}' already exists.")
    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles.")

    ok, err, audio, sr = validate_and_load_reference_audio(audio_path)
    if not ok:
        raise ValueError(f"Invalid reference audio: {err}")

    profile_id = str(uuid.uuid4())
    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    dest = profile_dir / "reference.wav"
    save_audio(audio, str(dest), sr)

    record = {
        "id": profile_id,
        "name": name,
        "engine": engine,
        "language": language,
        "voice_type": "cloned",
        "reference_audio": str(dest),
        "reference_text": reference_text or "",
        "preset_engine": None,
        "preset_voice_id": None,
        "created_at": _now(),
    }
    items = _load_index()
    items.insert(0, record)
    _save_index(items)
    meta_path = profile_dir / "metadata.json"
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def create_preset_profile(
    name: str,
    engine: str,
    preset_voice_id: str,
    language: str = "en",
) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise ValueError("Profile name is required.")
    if engine not in PRESET_ENGINES:
        raise ValueError(f"Engine '{engine}' is not a preset-voice engine.")
    existing = get_profile_by_name(name)
    if existing:
        return existing

    profile_id = str(uuid.uuid4())
    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": profile_id,
        "name": name,
        "engine": engine,
        "language": language,
        "voice_type": "preset",
        "reference_audio": None,
        "reference_text": "",
        "preset_engine": engine,
        "preset_voice_id": preset_voice_id,
        "created_at": _now(),
    }
    items = _load_index()
    items.insert(0, record)
    _save_index(items)
    (profile_dir / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def delete_profile(profile_id: str) -> bool:
    items = _load_index()
    kept = [p for p in items if p["id"] != profile_id]
    if len(kept) == len(items):
        return False
    _save_index(kept)
    profile_dir = config.get_profiles_dir() / profile_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir, ignore_errors=True)
    return True


def profile_choices() -> list[tuple[str, str]]:
    """Gradio dropdown choices: (label, id)."""
    out = []
    for p in list_profiles():
        kind = "preset" if p.get("voice_type") == "preset" else "cloned"
        label = f"{p['name']}  ·  {p.get('language', '?')}  ·  {kind}"
        out.append((label, p["id"]))
    return out


def voice_prompt_for_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Build the dict backends.generate() expects, matching Voicebox patterns."""
    voice_type = profile.get("voice_type") or "cloned"
    if voice_type == "preset":
        return {
            "voice_type": "preset",
            "preset_engine": profile.get("preset_engine"),
            "preset_voice_id": profile.get("preset_voice_id"),
        }
    ref = profile.get("reference_audio")
    if not ref or not Path(ref).exists():
        raise ValueError(f"Cloned profile '{profile.get('name')}' has no reference audio.")
    return {
        "ref_audio": str(ref),
        "ref_text": profile.get("reference_text") or "",
        "voice_type": "cloned",
    }
