
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

from ui.constants import active_slug, max_configs, max_name_length


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configurations"

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-")
    if slug:  return slug
    else:  return f"config_{max_configs}"

def ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def save_config(name: str, values: dict[str, str]) -> Path:
    ensure_dir()
    base_name = name.strip()
    if base_name.lower().endswith(".json"):
        base_name = base_name[: -len(".json")]
    if len(base_name) > max_name_length:
        raise ValueError(
            f"Config name must be {max_name_length} characters or fewer."
        )
    slug = _slugify(base_name)
    path = CONFIG_DIR / f"{slug}.json"
    if slug != active_slug and not path.exists():
        saved = [
            p for p in CONFIG_DIR.glob("*.json") if p.stem != active_slug]
        if len(saved) >= max_configs:
            raise ValueError(
                f"Maximum of {max_configs} saved configs reached. "
                "Delete one before saving another."
            )
    payload = {
        "name": base_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "values": values,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

def load_config(path: Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("values", {})

def delete_config(path: Path) -> None:
    Path(path).unlink(missing_ok=True)

def list_configs() -> list[Path]:
    ensure_dir()
    return sorted(
        CONFIG_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
