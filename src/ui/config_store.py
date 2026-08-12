
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re


CONFIG_DIR = Path(__file__).resolve().parents[2] / "dashboard_configs"

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-")
    return slug or "config"

def ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def save_config(name: str, values: dict[str, str]) -> Path:
    ensure_dir()
    slug = _slugify(name)
    path = CONFIG_DIR / f"{slug}.json"
    payload = {
        "name": name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "values": values,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

def load_config(path: Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("values", {})

def list_configs() -> list[Path]:
    ensure_dir()
    return sorted(
        CONFIG_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
