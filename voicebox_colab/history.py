"""Session generation history. Stored only under the Colab data dir."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config


def _load() -> list[dict[str, Any]]:
    path = config.get_history_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(items: list[dict[str, Any]]) -> None:
    config.get_history_path().write_text(json.dumps(items, indent=2), encoding="utf-8")


def add_entry(
    *,
    profile_name: str,
    engine: str,
    language: str,
    expression: str,
    text: str,
    audio_path: str,
    duration: float,
    model_name: str,
) -> dict[str, Any]:
    entry = {
        "id": str(uuid.uuid4()),
        "profile_name": profile_name,
        "engine": engine,
        "language": language,
        "expression": expression or "—",
        "text": text,
        "text_preview": (text[:80] + "…") if len(text) > 80 else text,
        "audio_path": audio_path,
        "duration": round(float(duration), 2),
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    items = _load()
    items.insert(0, entry)
    _save(items[:200])
    return entry


def list_entries() -> list[dict[str, Any]]:
    return _load()


def get_entry(entry_id: str) -> Optional[dict[str, Any]]:
    for item in _load():
        if item["id"] == entry_id:
            return item
    return None


def delete_entry(entry_id: str) -> bool:
    items = _load()
    kept = []
    deleted = None
    for item in items:
        if item["id"] == entry_id:
            deleted = item
        else:
            kept.append(item)
    if deleted is None:
        return False
    _save(kept)
    audio = Path(deleted.get("audio_path") or "")
    if audio.exists():
        try:
            audio.unlink()
        except OSError:
            pass
    return True


def history_labels() -> list[tuple[str, str]]:
    labels = []
    for i, item in enumerate(list_entries(), start=1):
        label = (
            f"{i:02d}  {item.get('profile_name', '?')} — "
            f"{item.get('language', '?')} — {item.get('expression', '—')}"
        )
        labels.append((label, item["id"]))
    return labels
