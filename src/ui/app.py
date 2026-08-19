
from __future__ import annotations

from nicegui import app, ui
from pathlib import Path
import subprocess
import sys

from ui import theme
from ui.components import render_field, section_card
from ui.config_store import (
    delete_config, list_configs, load_config, save_config,
)
from ui.constants import (
    active_slug, colors, group_icons, groups, max_name_length, nav_items
)
from ui.hp_schema import defaults_as_strings, fields_by_group

import run_config as rc


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
PIPELINE_SCRIPT = SRC_DIR / "pipeline.py"

def build_page() -> None:
    theme.apply()

    state = {
        "values": defaults_as_strings(),
        "errors": {},
        "process": None,  # subprocess.Popen | None
    }

    def _status_text() -> str:
        n_err = len(state["errors"])
        return "All fields valid" if n_err == 0 else f"{n_err} fields invalid"

    with ui.header().classes(
        "gp-titlebar items-center justify-between px-4 py-2"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.icon("stadium", size="24px").classes("text-purple-400")
            with ui.column().classes("gap-0"):
                ui.label("Gen Play — Control Panel")\
                    .classes("text-base font-semibold")
                ui.label("Hyperparameter dashboard")\
                    .classes("text-xs text-gray-500")
        with ui.row().classes("items-center gap-2"):
            status_pill = ui.label().classes("gp-pill gp-pill-muted")
            active_pill = ui.label().classes("gp-pill gp-pill-muted")
            run_pill = ui.label().classes("gp-pill gp-pill-muted")

    with ui.left_drawer(fixed=True).classes("gp-sidebar p-3").props(
        "width=220"
    ):
        ui.label("NAVIGATION")\
            .classes("text-xs text-gray-500 px-2 mb-1 tracking-wider")
        for label, icon, enabled in nav_items:
            classes = "gp-nav-item px-3 py-2 items-center gap-3 w-full"
            if enabled:  classes += " gp-nav-item-active"
            else:  classes += " cursor-not-allowed opacity-50"
            with ui.row().classes(classes):
                ui.icon(icon, size="18px")
                ui.label(label).classes("text-sm")
                if not enabled:
                    ui.space()
                    ui.label("soon").classes("text-[10px] text-gray-500")

    @ui.refreshable
    def form_grid() -> None:
        grouped_fields = fields_by_group()
        with ui.grid(columns=2).classes("w-full gap-4"):
            for group_name in groups:
                hp_fields = grouped_fields.get(group_name, [])
                if not hp_fields:
                    continue
                with section_card(
                    group_name, group_icons.get(group_name, "tune")
                ):
                    for hp_field in hp_fields:
                        render_field(
                            hp_field,
                            state["values"], state["errors"], _on_field_change
                        )

    def _refresh_active_pill() -> None:
        if rc.ACTIVE_CONFIG_PATH.exists():
            import json as _json

            try:
                payload = _json.loads(
                    rc.ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
                saved_at = payload.get("saved_at", "?")
            except (OSError, ValueError):
                saved_at = "?"
            active_pill.set_text(
                f"{rc.ACTIVE_CONFIG_PATH.name} set ({saved_at})")
            active_pill.style(f"color: {colors['success']}")
        else:
            active_pill.set_text(
                f"no {rc.ACTIVE_CONFIG_PATH.name} — "
                "main.py uses built-in defaults")
            active_pill.style(f"color: {colors['text_muted']}")

    def _on_field_change() -> None:
        status_pill.set_text(_status_text())
        status_pill.style(
            f"color: {
                colors['danger'] if state['errors'] else colors['success']
            }"
        )
        json_preview.set_text(_format_json())

    def _format_json() -> str:
        import json

        typed: dict[str, object] = {}
        for key, raw in state["values"].items():
            typed[key] = raw
        return json.dumps(typed, indent=2)

    with ui.column().classes("w-full max-w-5xl mx-auto p-6 gap-4"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Hyperparameters").classes("text-xl font-bold")
            with ui.row().classes("gap-2"):
                ui.button(
                    "Close", icon="power_settings_new",
                    on_click=lambda: _close_and_end_run(),
                ).props("outline color=negative dense").tooltip(
                    "Closes this dashboard. Ends run if in progress."
                )
                ui.button(
                    "Reset Defaults", icon="restart_alt",
                    on_click=lambda: _reset()
                ).props("outline color=orange dense")
                ui.button(
                    "Load", icon="folder_open",
                    on_click=lambda: load_dialog.open()
                ).props("outline color=primary dense")
                ui.button(
                    "Save", icon="save",
                    on_click=lambda: save_dialog.open()
                ).props("color=primary dense")
                ui.button(
                    "Set Active", icon="bolt", on_click=lambda: _set_active()
                ).props("color=blue dense").tooltip(
                    f"Writes these values to configurations/"
                    f"{rc.ACTIVE_CONFIG_PATH.name} — pipeline.py loads this "
                    "on next run. Only non-invalid fields are allowed."
                )
                run_button = (
                    ui.button(
                        "Start", icon="rocket_launch",
                        on_click=lambda: _start_run()
                    )
                    .props("color=positive dense")
                    .tooltip(
                        f"Saves as {rc.ACTIVE_CONFIG_PATH.name}, runs "
                        "`python pipeline.py` as its own process, then "
                        "closes this dashboard."
                    )
                )

        form_grid()

        with ui.expansion("Live JSON preview", icon="data_object").classes(
            "gp-card w-full"
        ):
            json_preview = ui.label(_format_json()).classes(
                "text-xs font-mono whitespace-pre-wrap text-gray-300"
            )

    with ui.dialog() as save_dialog, ui.card().classes(
        "gp-card p-4 gap-3 w-96"
    ):
        ui.label("Save configuration").classes("text-base font-semibold")
        name_input = ui.input(
            "Config name", value="my-config"
        ).props(
            f"outlined dense maxlength={max_name_length} counter"
        ).classes("w-full")
        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button(
                "Cancel", on_click=save_dialog.close).props("flat dense")

            def _do_save() -> None:
                try:
                    path = save_config(
                        name_input.value or "config", state["values"])
                except ValueError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                ui.notify(f"Saved -> {path.name}", type="positive")
                save_dialog.close()
                refresh_load_list.refresh()

            ui.button("Save", on_click=_do_save).props("color=primary dense")

    with ui.dialog() as load_dialog, ui.card().classes(
        "gp-card p-4 gap-3 w-96"
    ):
        ui.label("Load configuration").classes("text-base font-semibold")
        _load_dialog_state = {"selected": None}

        def _config_row(cfg_path: Path) -> None:
            is_selected = _load_dialog_state["selected"] == cfg_path
            with ui.row().classes(
                "items-center justify-between w-full gp-nav-item px-2 py-1"
            ):
                with ui.row().classes("items-center gap-2"):
                    def _load(p=cfg_path) -> None:
                        loaded = load_config(p)
                        state["values"].update(loaded)
                        state["errors"].clear()
                        form_grid.refresh()
                        _on_field_change()
                        _load_dialog_state["selected"] = p
                        load_dialog.close()
                        ui.notify(f"Loaded {p.stem}", type="info")

                    ui.button(
                        icon=(
                            "radio_button_checked" if is_selected
                            else "radio_button_unchecked"
                        ),
                        on_click=_load,
                    ).props("flat dense round size=sm color=green")
                    ui.label(cfg_path.stem).classes("text-sm")

                def _delete(p=cfg_path) -> None:
                    delete_config(p)
                    if _load_dialog_state["selected"] == p:
                        _load_dialog_state["selected"] = None
                    refresh_load_list.refresh()
                    ui.notify(f"Deleted {p.stem}", type="warning")

                ui.button(
                    icon="delete", on_click=_delete
                ).props("flat dense round size=sm color=negative")

        @ui.refreshable
        def refresh_load_list() -> None:
            configs = list_configs()
            if not configs:
                ui.label(
                    "No saved configs yet."
                ).classes("text-xs text-gray-500")
            for cfg_path in configs:
                _config_row(cfg_path)

        with ui.column().classes("w-full gap-1 max-h-80 overflow-auto"):
            refresh_load_list()

    def _reset() -> None:
        state["values"] = defaults_as_strings()
        state["errors"] = {}
        form_grid.refresh()
        _on_field_change()

    def _save_active() -> bool:
        if state["errors"]:
            ui.notify(
                f"Fix {len(state['errors'])} invalid fields.",
                type="negative",
            )
            return False
        save_config(active_slug, state["values"])
        _refresh_active_pill()
        refresh_load_list.refresh()
        return True

    def _set_active() -> None:
        if _save_active():
            ui.notify(
                f"Saved to configurations/{rc.ACTIVE_CONFIG_PATH.name} — "
                "pipeline.py will use these values on its next run.",
                type="positive",
            )

    def _refresh_run_pill() -> None:
        proc = state["process"]
        if proc is None:
            run_pill.set_text("idle")
            run_pill.style(f"color: {colors['text_muted']}")
            run_button.enable()
        elif proc.poll() is None:
            run_pill.set_text(f"running (pid {proc.pid})")
            run_pill.style(f"color: {colors['warning']}")
            run_button.disable()
        else:
            code = proc.returncode
            ok = code == 0
            run_pill.set_text(f"last run exited {code}")
            run_pill.style(
                f"color: {
                    colors['success'] if ok else colors['danger']
                }"
            )
            run_button.enable()

    def _poll_run() -> None:
        proc = state["process"]
        if proc is None:
            return
        if proc.poll() is not None:
            code = proc.returncode
            ui.notify(
                f"pipeline.py finished (exit {code})",
                type="positive" if code == 0 else "negative",
            )
        _refresh_run_pill()

    ui.timer(2.0, _poll_run)

    def _terminate_run() -> None:
        proc = state["process"]
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _close_dashboard() -> None:
        app.shutdown()

    def _close_and_end_run() -> None:
        _terminate_run()
        _close_dashboard()

    def _start_run() -> None:
        proc = state["process"]
        if proc is not None and proc.poll() is None:
            ui.notify("A run is already in progress.", type="warning")
            return
        if not _save_active():
            return
        state["process"] = subprocess.Popen(
            [sys.executable, str(PIPELINE_SCRIPT)], cwd=str(PROJECT_ROOT),
        )
        _refresh_run_pill()
        ui.notify(
            f"Started pipeline.py (pid {state['process'].pid}) — "
            "watch the terminal for progress. Closing dashboard...",
            type="info",
        )
        ui.timer(1.0, _close_dashboard, once=True)

    _on_field_change()
    _refresh_active_pill()
    _refresh_run_pill()
