
from __future__ import annotations

from nicegui import ui # pyright: ignore[reportMissingImports]
from typing import Callable

from common.constants import fps
from run_config import parse_int_list
from ui.hp_schema import (
    HPField,
    catg_idxs_to_labels, catg_labels_to_idxs, catg_label_tokens_valid
)


_EXCLUSIVE_EPS = 1e-6

_CLAMP_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "dropout": (0.0, 1.0, False),
    "dim_play": (1, float("inf"), False),
    "dim_player": (1, float("inf"), False),
    "final_dim": (1, float("inf"), False),
    "diff_eq_dim_middle": (1, float("inf"), False),
    "drift_size": (2, 4, False),
    "diffusion_size": (0, 6, False),
    "pow_iters": (1, 4, False),
    "batch_size": (1, float("inf"), False),
    "num_epochs": (1, float("inf"), False),
    "n_parallel": (1, float("inf"), False),
    "n_trials": (1, float("inf"), False),
    "decoder_min_stdv": (1e-5, 1e-1, False),
    "decoder_max_stdv": (1e-5, 1e-1, False),
}

def _clamp(key: str, value: float) -> float:
    lo, hi, exclusive = _CLAMP_BOUNDS[key]
    if exclusive:
        lo = lo + _EXCLUSIVE_EPS
        hi = hi - _EXCLUSIVE_EPS
    return min(max(value, lo), hi)

_PAIRED_BOUNDS: dict[str, tuple[str, str]] = {
    "decoder_min_stdv": ("decoder_max_stdv", "lower"),
    "decoder_max_stdv": ("decoder_min_stdv", "upper"),
}

def _resolve_pair(key: str, value: float, values: dict[str, str]) -> float:
    sibling_key, role = _PAIRED_BOUNDS[key]
    try:
        sibling = float(values.get(sibling_key, ""))
    except ValueError:
        return value
    if role == "lower" and value >= sibling:
        return _clamp(key, sibling - _EXCLUSIVE_EPS)
    if role == "upper" and value <= sibling:
        return _clamp(key, sibling + _EXCLUSIVE_EPS)
    return value

_LIST_CLAMP_BOUNDS: dict[str, tuple[int, int]] = {
    "lags_def_tm": (0, fps * 30),
    "lags_off_tm": (0, fps * 30),
    "lags_def_op": (0, fps * 30),
    "lags_off_op": (0, fps * 30),
}

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
                if ok and key in _LIST_CLAMP_BOUNDS:
                    lo, hi = _LIST_CLAMP_BOUNDS[key]
                    ok, result = _clamp_int_list(raw, lo, hi)
                    if ok and result != raw:
                        raw = result
                        text_input.set_value(raw)
                if ok and key in _CLAMP_BOUNDS:
                    clamped = _clamp(key, result)
                    if clamped != result:
                        result = clamped
                        raw = str(result)
                        text_input.set_value(raw)
                if ok and key in _PAIRED_BOUNDS:
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
