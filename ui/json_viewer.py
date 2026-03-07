"""
Aurora JSON Dropdown Viewer — ui/json_viewer.py
================================================
Streamlit component providing a multi-mode JSON viewer with dropdown selector.

Modes:
  - Full JSON:       Complete JSON tree with expand/collapse
  - Top-level only:  Scalar fields only, nested structures summarized
  - Table view:      Read-only dataframe of line items (when tabular_state exists)
  - Raw:             Raw JSON string with syntax highlighting

RULES:
  - Uses :material/ icons only (no emoji)
  - Provides aria-labels via Streamlit's built-in accessibility
  - Does NOT modify master_json or any session state
"""

import json as _json
import pandas as pd
import streamlit as st
from typing import Any, Optional


def render_json_viewer(
    json_data: Any,
    tabular_state_exists: bool = False,
    table_rows: Optional[list] = None,
    container_height: int = 720,
    key_prefix: str = "json_viewer",
):
    """
    Renders a multi-mode JSON viewer with a dropdown mode selector.

    Args:
        json_data:             The JSON data to display (dict or list)
        tabular_state_exists:  Whether tabular mode is active (enables Table view option)
        table_rows:            Rows for table view (from tabular_preview or master_df)
        container_height:      Height of the viewer container in pixels
        key_prefix:            Unique key prefix for Streamlit widgets
    """
    if json_data is None:
        st.info(":material/info: No JSON data available. Process a document first.")
        return

    # ── Mode Selector Header ──────────────────────────────────────────────
    mode_options = ["Full JSON", "Top-level only", "Raw"]
    if tabular_state_exists:
        mode_options.insert(2, "Table view")

    hdr_col1, hdr_col2 = st.columns([1, 2])
    with hdr_col1:
        st.markdown(":material/code: **JSON Viewer**")
    with hdr_col2:
        view_mode = st.selectbox(
            "View mode",
            mode_options,
            index=0,
            key=f"{key_prefix}_mode",
            label_visibility="collapsed",
        )

    # ── Render Selected Mode ──────────────────────────────────────────────
    with st.container(height=container_height):
        if view_mode == "Full JSON":
            _render_full_json(json_data)

        elif view_mode == "Top-level only":
            _render_top_level(json_data)

        elif view_mode == "Table view":
            _render_table_view(json_data, table_rows)

        elif view_mode == "Raw":
            _render_raw(json_data)


# ── Mode Renderers ────────────────────────────────────────────────────────────

def _render_full_json(data: Any):
    """Full JSON tree with built-in expand/collapse."""
    st.json(data)


def _render_top_level(data: Any):
    """Shows only scalar fields; nested structures are summarized."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                st.markdown(
                    f"**{key}** :material/list: `array` — {len(value)} items"
                )
            elif isinstance(value, dict):
                st.markdown(
                    f"**{key}** :material/data_object: `object` — {len(value)} keys"
                )
            else:
                st.markdown(f"**{key}**: {value}")
    elif isinstance(data, list):
        st.markdown(f":material/list: Array with **{len(data)}** items")
        if data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            st.caption(f"Fields: {', '.join(keys[:10])}{'...' if len(keys) > 10 else ''}")
    else:
        st.write(data)


def _render_table_view(data: Any, table_rows: Optional[list] = None):
    """Read-only interactive table preview of line items."""
    rows = table_rows

    # If no explicit rows provided, try to extract from JSON
    if not rows:
        if isinstance(data, dict):
            # Find the first array-of-objects in the JSON
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    rows = value
                    st.caption(f":material/table_chart: Showing **{key}** ({len(rows)} rows)")
                    break
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            rows = data

    if rows:
        df = pd.DataFrame(rows)
        # Hide internal metadata columns
        hidden = {"__source_row_id", "__master_path", "__created_at_confidence", "__row_version"}
        visible_cols = [c for c in df.columns if c not in hidden]
        st.dataframe(
            df[visible_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(":material/info: No tabular data available for table view.")


def _render_raw(data: Any):
    """Raw JSON string with syntax highlighting."""
    try:
        raw_str = _json.dumps(data, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        raw_str = str(data)
    st.code(raw_str, language="json")
