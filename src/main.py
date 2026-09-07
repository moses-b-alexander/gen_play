
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nicegui import ui  # noqa: E402
import wsproto.utilities  # noqa: E402

from ui.app import build_page  # noqa: E402


@ui.page("/")
def index() -> None:
    build_page()


if __name__ in {"__main__", "__mp_main__"}:
    try:
        import webview  # noqa: F401
        native = True
    except ImportError:
        native = False

    try:
        ui.run(
            title="Gen Play — Control Panel",
            native=native, window_size=(1360, 900) if native else None,
            reload=False, show=True,
            favicon="🏈"
        )
    except wsproto.utilities.LocalProtocolError as exc:
        if not (
            "CloseConnection" in str(exc) and
            "ConnectionState.CLOSED" in str(exc)
        ):
            raise
    except asyncio.CancelledError:
        pass
