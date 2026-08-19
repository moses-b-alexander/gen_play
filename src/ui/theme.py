
from __future__ import annotations

from nicegui import ui

from ui.constants import colors


_CSS = f"""
:root {{
    --gp-bg: {colors["bg"]};
    --gp-bg-alt: {colors["bg_alt"]};
    --gp-surface: {colors["surface"]};
    --gp-surface-hover: {colors["surface_hover"]};
    --gp-border: {colors["border"]};
    --gp-accent: {colors["accent"]};
    --gp-accent-soft: {colors["accent_soft"]};
    --gp-text: {colors["text"]};
    --gp-text-muted: {colors["text_muted"]};
    --gp-success: {colors["success"]};
    --gp-danger: {colors["danger"]};
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
        primary=colors["accent"],
        positive=colors["success"], negative=colors["danger"]
    )
