
from __future__ import annotations

from nicegui import ui # pyright: ignore[reportMissingImports]


COLORS = {
    "bg": "#14151b",
    "bg_alt": "#191a22",
    "surface": "#1d1f29",
    "surface_hover": "#242631",
    "border": "#2b2e3a",
    "accent": "#7c5cff",
    "accent_soft": "#7c5cff26",
    "text": "#e7e8ee",
    "text_muted": "#9297ab",
    "success": "#34d399",
    "danger": "#f87171",
    "warning": "#fbbf24",
}

_CSS = f"""
:root {{
    --gp-bg: {COLORS["bg"]};
    --gp-bg-alt: {COLORS["bg_alt"]};
    --gp-surface: {COLORS["surface"]};
    --gp-surface-hover: {COLORS["surface_hover"]};
    --gp-border: {COLORS["border"]};
    --gp-accent: {COLORS["accent"]};
    --gp-accent-soft: {COLORS["accent_soft"]};
    --gp-text: {COLORS["text"]};
    --gp-text-muted: {COLORS["text_muted"]};
    --gp-success: {COLORS["success"]};
    --gp-danger: {COLORS["danger"]};
}}

body {{
    background: var(--gp-bg) !important;
    color: var(--gp-text);
    font-family: 'Segoe UI', Inter, -apple-system, sans-serif;
}}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: var(--gp-border); border-radius: 8px;
}}
::-webkit-scrollbar-thumb:hover {{ background: var(--gp-surface-hover); }}

.gp-titlebar {{
    background: var(--gp-bg-alt);
    border-bottom: 1px solid var(--gp-border);
}}

.gp-sidebar {{
    background: var(--gp-bg-alt);
    border-right: 1px solid var(--gp-border);
}}

.gp-nav-item {{
    border-radius: 8px;
    transition: background 120ms ease, color 120ms ease;
    color: var(--gp-text-muted);
}}
.gp-nav-item:hover {{
    background: var(--gp-surface-hover);
    color: var(--gp-text);
}}
.gp-nav-item-active {{
    background: var(--gp-accent-soft);
    color: var(--gp-accent) !important;
    font-weight: 600;
}}

.gp-card {{
    background: var(--gp-surface);
    border: 1px solid var(--gp-border);
    border-radius: 14px;
    transition: border-color 120ms ease;
}}
.gp-card:hover {{
    border-color: #3a3d4d;
}}

.gp-field-input .q-field__control {{
    background: var(--gp-bg-alt) !important;
    border-radius: 8px !important;
}}
.gp-field-input .q-field__native {{
    color: var(--gp-text) !important;
}}
.gp-field-input.gp-field-invalid .q-field__control {{
    box-shadow: 0 0 0 1px var(--gp-danger) inset;
}}

.gp-pill {{
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .02em;
}}
.gp-pill-muted {{
    background: #2b2e3a;
    color: var(--gp-text-muted);
}}

.gp-scrollarea {{
    scrollbar-gutter: stable;
}}
"""

def apply() -> None:
    ui.dark_mode(True)
    ui.add_head_html(f"<style>{_CSS}</style>")
    ui.colors(
        primary=COLORS["accent"],
        positive=COLORS["success"], negative=COLORS["danger"]
    )
