
from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nicegui import ui  # noqa: E402

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

    # app.shutdown() races uvicorn's wsproto websocket teardown when the
    # browser's connection has already closed itself.
    logging.getLogger("uvicorn.error").addFilter(
        lambda record: "CloseConnection" not in record.getMessage()
    )

    ui.run(
        title="Gen Play — Control Panel",
        native=native,
        window_size=(1360, 900) if native else None,
        reload=False,
        show=True,
        favicon="🏈",
    )
