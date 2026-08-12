
from __future__ import annotations

from nicegui import ui # pyright: ignore[reportMissingImports]
from typing import Callable

from ui.hp_schema import HPField


def cast_value(field: HPField, raw: str) -> tuple[bool, object | str]:
    raw = raw.strip()
    if field.kind == "bool":
        if raw.lower() in ("true", "1", "yes", "y", "on"):
            return (True, True)
        if raw.lower() in ("false", "0", "no", "n", "off"):
            return (True, False)
        return (False, f"expected true/false, got '{raw}'")
    if field.kind == "int":
        try:
            return (True, int(raw))
        except ValueError:
            return (False, f"expected an integer, got '{raw}'")
    if field.kind == "float":
        try:
            return (True, float(raw))
        except ValueError:
            return False, f"expected a number, got '{raw}'"
    return (True, raw)

def render_field(
    hp_field: HPField,
    values: dict[str, str],
    errors: dict[str, str],
    on_change: Callable[[], None],
) -> None:
    with ui.column().classes("gap-1 w-full"):
        with ui.row().classes("items-center gap-2"):
            ui.label(hp_field.label).classes("text-sm font-medium")
            if hp_field.suggested_values:
                suggestions = \
                    ", ".join(str(v) for v in hp_field.suggested_values)
                ui.icon("info", size="16px").classes("text-gray-500").tooltip(
                    f"Suggested: {suggestions}"
                )

        text_input = (
            ui.input(value=values.get(hp_field.key, str(hp_field.default)))
            .props("outlined dense")
            .classes("gp-field-input w-full")
        )
        if hp_field.description:
            ui.label(hp_field.description)\
                .classes("text-xs text-gray-500 leading-tight")

        error_label = ui.label("").classes("text-xs text-red-400")

        def _refresh_error_style() -> None:
            if hp_field.key in errors:
                text_input.classes(add="gp-field-invalid")
                error_label.set_text(errors[hp_field.key])
            else:
                text_input.classes(remove="gp-field-invalid")
                error_label.set_text("")

        def _handle_change(e, key=hp_field.key, f=hp_field) -> None:
            raw = "" if e.value is None else str(e.value)
            values[key] = raw
            ok, result = cast_value(f, raw)
            if ok:
                errors.pop(key, None)
            else:
                errors[key] = str(result)
            _refresh_error_style()
            on_change()

        text_input.on_value_change(_handle_change)
        _refresh_error_style()

def section_card(title: str, icon: str = "tune"):
    card = ui.card().classes("gp-card w-full p-5 gap-3")
    with card:
        with ui.row().classes("items-center gap-2 mb-1"):
            ui.icon(icon, size="20px").classes("text-purple-400")
            ui.label(title).classes("text-base font-semibold")
        ui.separator().classes("opacity-10")
    return card
