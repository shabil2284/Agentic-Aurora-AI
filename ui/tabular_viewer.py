"""
Aurora Tabular Viewer — ui/tabular_viewer.py
============================================
Streamlit component for the editable Tabular Mode panel.
Renders editable grid, Apply / Revert buttons, Export controls,
and SQL Sync two-step commit.

IMPORTANT: This module does NOT import from app.py or alter session_state
keys outside of what is documented here:
  - st.session_state.tabular_staged_rows  (editable grid state)
  - st.session_state.tabular_table_ready  (SQL table created flag)
  - st.session_state.tabular_autosync_on  (AutoSync enabled flag)
  - st.session_state.tabular_conflicts    (conflict list from last upsert)
"""

import io
import json as _json
import pandas as pd
import streamlit as st

from utils.tabularizer   import Tabularizer
from utils.sql_sync_adapter import SQLSyncAdapter

# ── Columns hidden from the editable grid (but preserved in rows) ─────────────
HIDDEN_COLS = {"__source_row_id", "__master_path", "__created_at_confidence", "__row_version"}


def _visible_df(rows: list) -> pd.DataFrame:
    """Build a DataFrame with only user-visible columns."""
    df = pd.DataFrame(rows)
    drop = [c for c in df.columns if c in HIDDEN_COLS]
    return df.drop(columns=drop, errors="ignore")


def _merge_edits(original_rows: list, edited_df: pd.DataFrame) -> list:
    """
    Merge the user's edited cells back into the full rows list that still
    carries the hidden metadata columns.
    """
    merged = []
    for idx, orig_row in enumerate(original_rows):
        updated = dict(orig_row)   # start from original (includes meta)
        if idx < len(edited_df):
            for col in edited_df.columns:
                updated[col] = edited_df.iloc[idx][col]
        merged.append(updated)
    return merged


# ── Main Entry Point ──────────────────────────────────────────────────────────

def render_tabular_viewer(tabular_preview: dict, master_json):
    """
    Renders the complete Tabular Mode panel.
    Called from app.py only when toggle is ON and confidence >= 0.85.
    """
    confidence   = tabular_preview.get("confidence", 0)
    doc_type     = tabular_preview.get("doc_type", "generic_table")
    preview_only = tabular_preview.get("preview_only", False)
    
    # ── Table View MUST read from: table_dataset (master_df) NOT cached_table ──
    _mdf = st.session_state.get("master_df")
    if _mdf is not None and not _mdf.empty:
        rows = _mdf.to_dict(orient="records")
    else:
        rows = tabular_preview.get("rows", [])

    # ── Init scoped session keys and Live Sync Event Listener ─────────────────
    _current_global_event = st.session_state.get("MASTER_DATA_UPDATED_EVENT", 0)

    if (
        "tabular_staged_rows" not in st.session_state or
        st.session_state.get("_tabular_last_global_event", -1) != _current_global_event
    ):
        st.session_state.tabular_staged_rows    = list(rows)
        st.session_state._tabular_last_global_event = _current_global_event
    if "tabular_table_ready" not in st.session_state:
        st.session_state.tabular_table_ready = False
    if "tabular_autosync_on" not in st.session_state:
        st.session_state.tabular_autosync_on = False
    if "tabular_conflicts" not in st.session_state:
        st.session_state.tabular_conflicts = []

    staged_rows = st.session_state.tabular_staged_rows

    # ── Header ────────────────────────────────────────────────────────────────
    st.divider()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.subheader(":material/table_chart: Tabular Mode")
        st.caption(f"Doc type: **{doc_type}** · Confidence: **{confidence:.2f}** · {len(staged_rows)} rows")
    with col_b:
        if preview_only:
            st.warning(f"Preview capped at {len(staged_rows):,} rows · click 'Generate Full Table' to load all")

    # ── Large-array guard ─────────────────────────────────────────────────────
    if preview_only:
        if not st.button(":material/warning: Generate Full Table (may be slow)", type="secondary"):
            return

    # ── Editable Grid ─────────────────────────────────────────────────────────
    visible_df = _visible_df(staged_rows)

    st.markdown("**Edit cells directly. Changes are staged until you click *Apply*.**")
    edited_df = st.data_editor(
        visible_df,
        use_container_width=True,
        num_rows="dynamic",          # allows row add/delete
        key="tabular_grid_editor",
    )

    # ── Action buttons row ────────────────────────────────────────────────────
    # hide local export controls here — exports accessible via global Export Data section
    btn_apply, btn_revert = st.columns(2)
    
    import pandas as pd

    # Apply to Master Model (table_dataset -> master_json serialization)
    with btn_apply:
        if st.button(":material/check_circle: Apply to master_json", type="primary", use_container_width=True):
            # Merge edits
            merged = _merge_edits(staged_rows, edited_df)
            st.session_state.tabular_staged_rows = merged

            _mj = st.session_state.get("master_json")
            _doc_mode = st.session_state.get("doc_mode", "TABLE_MODE")

            # ── HIERARCHICAL MODE: write-back via path bindings ──
            if (
                _doc_mode in ("HIERARCHICAL_MODE", "KEY_VALUE_MODE", "ENTITY_LIST_MODE")
                and isinstance(_mj, dict)
            ):
                import copy
                from utils.path_utils import find_array_key, set_by_path
                from utils.live_sync import refresh_all_views

                _updated_json = copy.deepcopy(_mj)
                _arr_key = find_array_key(_updated_json)

                if _arr_key and _arr_key in _updated_json:
                    # Rebuild child array from merged rows (strip parent scalars)
                    parent_keys = {k for k, v in _updated_json.items() if not isinstance(v, (list, dict))}
                    new_children = []
                    for r in merged:
                        child = {k: v for k, v in r.items() if k not in parent_keys and not str(k).startswith("__")}
                        new_children.append(child)
                    _updated_json[_arr_key] = new_children

                    st.session_state.master_json = _updated_json
                    st.session_state.json_data = _updated_json
                    refresh_all_views(source="TABULAR_VIEW_PATH_APPLY")
                else:
                    # No array key — fallback to flat update
                    _new_df = pd.DataFrame(merged)
                    from utils.live_sync import update_master_model
                    update_master_model(df=_new_df, source="TABULAR_VIEW_APPLY")
            else:
                # ── TABLE_MODE: standard df-based update ──
                _new_df = pd.DataFrame(merged)
                from utils.live_sync import update_master_model
                update_master_model(df=_new_df, source="TABULAR_VIEW_APPLY")

            st.session_state.canonical_data = st.session_state.master_json
            # Re-index vectors
            try:
                from vectorstore.session_vector_registry import SessionVectorRegistry
                import uuid as _uuid
                store = SessionVectorRegistry.get_store(st.session_state.session_id)
                store.add_documents(
                    [_json.dumps(st.session_state.master_json)],
                    [{"filename": st.session_state.get("current_file_path", "").split("/")[-1]}],
                    [str(_uuid.uuid4())]
                )
                st.toast("Vector index updated", icon=":material/check:")
                st.success(":material/check: Applied — master_json updated & all views refreshed.")
            except Exception as e:
                st.warning(f"Vector re-index skipped: {e}")
            st.rerun()

    # Revert
    with btn_revert:
        if st.button(":material/undo: Revert Changes", type="secondary", use_container_width=True):
            st.session_state.tabular_staged_rows = list(rows)
            st.rerun()
