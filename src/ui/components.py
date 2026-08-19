
from __future__ import annotations

from nicegui import ui
from typing import Callable

from run_config import parse_int_list
from ui.constants import (
    clamp_bounds, exclusive_eps, list_clamp_bounds, paired_bounds
)
from ui.hp_schema import (
    HPField,
    catg_idxs_to_labels, catg_labels_to_idxs, catg_label_tokens_valid
)


def _clamp(key: str, value: float) -> float:
    lo, hi, exclusive = clamp_bounds[key]
    if exclusive:
        lo = lo + exclusive_eps
        hi = hi - exclusive_eps
    return min(max(value, lo), hi)

def _resolve_pair(key: str, value: float, values: dict[str, str]) -> float:
    sibling_key, role = paired_bounds[key]
    try:
        sibling = float(values.get(sibling_key, ""))
    except ValueError:
        return value
    if role == "lower" and value >= sibling:
        return _clamp(key, sibling - exclusive_eps)
    if role == "upper" and value <= sibling:
        return _clamp(key, sibling + exclusive_eps)
    return value

def _clamp_int_list(raw: str, lo: int, hi: int) -> tuple[bool, str]:
    try:
        vals = parse_int_list(raw)
    except ValueError:
        return False, f"expected comma-separated integers, got '{raw}'"
    clamped = [min(max(v, lo), hi) for v in vals]
    return True, ",".join(str(v) for v in clamped)

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
    values: dict[str, str], errors: dict[str, str],
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

        initial_value = values.get(hp_field.key, str(hp_field.default))
        if hp_field.key == "catg_idxs":
            initial_value = catg_idxs_to_labels(initial_value)

        text_input = (
            ui.input(value=initial_value)
            .props("outlined dense")
            .classes("gp-field-input w-full")
        )
        if hp_field.description:
            ui.label(hp_field.description).classes(
                "text-xs text-gray-500 leading-tight whitespace-pre-line"
            )

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
            if key == "catg_idxs":
                values[key] = catg_labels_to_idxs(raw)
                ok = catg_label_tokens_valid(raw)
                result = f"unrecognized category in '{raw}'"
            else:
                ok, result = cast_value(f, raw)
                if ok and key in list_clamp_bounds:
                    lo, hi = list_clamp_bounds[key]
                    ok, result = _clamp_int_list(raw, lo, hi)
                    if ok and result != raw:
                        raw = result
                        text_input.set_value(raw)
                if ok and key in clamp_bounds:
                    clamped = _clamp(key, result)
                    if clamped != result:
                        result = clamped
                        raw = str(result)
                        text_input.set_value(raw)
                if ok and key in paired_bounds:
                    paired = _resolve_pair(key, result, values)
                    if paired != result:
                        result = paired
                        raw = str(result)
                        text_input.set_value(raw)
                values[key] = raw
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
