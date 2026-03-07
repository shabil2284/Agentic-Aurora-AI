"""
Aurora Live Sync Engine — utils/live_sync.py
=============================================
Single Source of Truth synchronization layer.

All edits (CSV Editor, SQL Engine, Schema Editor, Table View, JSON Viewer)
MUST go through this module to update the MASTER_DATA_MODEL and broadcast
changes to every component instantly.

RULES:
- In TABLE mode:  MASTER_DATA_MODEL = master_df (a pandas DataFrame)
- In JSON mode:   MASTER_DATA_MODEL = master_json (a dict/list)
- broadcast_sync() propagates the master to ALL session state aliases
  (df, csv_df, master_json, json_data) and the SQLite DB.
"""

import pandas as pd
import json
from typing import Any, Optional, List, Dict
import streamlit as st


def update_master_model(
    df: Optional[pd.DataFrame] = None,
    json_data: Optional[Any] = None,
    source: str = "unknown",
):
    """
    Single write entry point. Call this instead of directly setting
    st.session_state.master_df or st.session_state.master_json.

    Pass EITHER df (for table edits) OR json_data (for JSON edits).
    """
    if df is not None:
        st.session_state.master_df = df
        # Derive JSON from the DataFrame for JSON viewer sync
        try:
            _records = df.to_dict(orient="records")
            
            _old_json = st.session_state.get("master_json")
            if st.session_state.get("tabular_mode_on") and isinstance(_old_json, dict):
                import copy
                _new_json = copy.deepcopy(_old_json)
                
                # STEP 1: Extract MASTER_PARENT_OBJECT and find the array key
                from utils.tabularizer import TabularDetector, TABULAR_ARRAY_KEYS
                array_key = None
                for k in _new_json.keys():
                    if k.lower().strip() in TABULAR_ARRAY_KEYS and isinstance(_new_json[k], list):
                        array_key = k
                        break
                if array_key is None:
                    struct_hit = TabularDetector._find_repeating_arrays(_new_json, max_depth=2)
                    if struct_hit:
                        _sk, _sc, _skeys = struct_hit
                        if _sk in _new_json:
                            array_key = _sk
                
                if array_key:
                    # Identify primitives to strip from the flat table
                    parent_keys = {k for k, v in _new_json.items() if not isinstance(v, (list, dict))}
                    print(f"[LiveSync Debug] Found array_key: {array_key}, parent_keys: {parent_keys}")
                    
                    # STEP 2: Convert table_dataset -> NEW_CHILD_ARRAY
                    new_child_array = []
                    for r in _records:
                        child_obj = {}
                        for k, v in r.items():
                            # Keep if it isn't a parent scalar, or if the user explicitly added a column
                            if k not in parent_keys:
                                child_obj[k] = v
                        new_child_array.append(child_obj)
                        
                    # STEP 3: Rebuild master_json using MERGE
                    _new_json[array_key] = new_child_array
                    
                    st.session_state.master_json = _new_json
                    st.session_state.json_data   = _new_json
                else:
                    # Fallback: Instead of wiping the dict into a list, FORCE an array key
                    print("[LiveSync Debug] No array key found! Forcing 'Items' key to preserve parents.")
                    _new_json["Items"] = _records
                    st.session_state.master_json = _new_json
                    st.session_state.json_data   = _new_json
            else:
                st.session_state.master_json = _records
                st.session_state.json_data   = _records
            
            # Auto-derive schema so Schema Editor uses the live column list
            if st.session_state.get("tabular_mode_on"):
                from schema_generator import SchemaGenerator
                st.session_state.schema_list = SchemaGenerator.derive_auto(st.session_state.master_json, "TABLE_MODE")
        except Exception:
            pass
        print(f"[LiveSync] master_df updated from {source} ({len(df)} rows, {len(df.columns)} cols)")

    elif json_data is not None:
        st.session_state.master_json = json_data
        st.session_state.json_data   = json_data
        # Derive DataFrame from JSON for table/CSV viewer sync (if tabular mode)
        if st.session_state.get("tabular_mode_on"):
            try:
                if isinstance(json_data, list) and json_data and isinstance(json_data[0], dict):
                    _new_df = pd.DataFrame(json_data)
                    _visible = [c for c in _new_df.columns if not str(c).startswith("__")]
                    st.session_state.master_df = _new_df[_visible].copy()
                    
                    # Auto-derive schema so Schema Editor uses the live column list
                    from schema_generator import SchemaGenerator
                    st.session_state.schema_list = SchemaGenerator.derive_auto(json_data, "TABLE_MODE")
                    print(f"[LiveSync] master_df derived from JSON ({len(_new_df)} rows)")
            except Exception:
                pass
        print(f"[LiveSync] master_json updated from {source}")

    # Always broadcast after update
    broadcast_sync(source=source)


def broadcast_sync(source: str = "unknown"):
    """
    Propagates the current MASTER_DATA_MODEL to ALL aliases and the DB.
    Call this after any update to ensure zero desync.
    """
    _mdf = st.session_state.get("master_df")

    if _mdf is not None and not _mdf.empty:
        # ── Propagate DataFrame to all aliases ──────────────────────────
        st.session_state.df     = _mdf
        st.session_state.csv_df = _mdf

        # ── Sync to SQLite DB for SQL Engine ────────────────────────────
        _active_tbl = (
            st.session_state.get("active_table")
            or st.session_state.get("orch_target")
            or "document"
        )
        try:
            from utils.sync_engine import SyncEngine
            SyncEngine.sync_all(_mdf, _active_tbl)
        except Exception as e:
            print(f"[LiveSync] DB sync warning: {e}")

    # ── JSON always stays in sync ───────────────────────────────────────
    _mj = st.session_state.get("master_json")
    if _mj is not None:
        st.session_state.json_data = _mj

    # ── Fire Global Update Event ─────────────────────────────────────────
    # Schema Editor and Table View will listen to this event. If it increments,
    # they must drop their caches and re-read from the LIVE MASTER_DATA_MODEL.
    if "MASTER_DATA_UPDATED_EVENT" not in st.session_state:
        st.session_state.MASTER_DATA_UPDATED_EVENT = 0
    st.session_state.MASTER_DATA_UPDATED_EVENT += 1
    print(f"[LiveSync] Fired MASTER_DATA_UPDATED_EVENT: {st.session_state.MASTER_DATA_UPDATED_EVENT} (Source: {source})")


def update_schema(schema_list: List[Dict], source: str = "schema_editor"):
    """
    Applies schema changes (column renames, deletes, type changes) to
    the MASTER_DATA_MODEL and broadcasts.
    """
    st.session_state.schema_list = schema_list

    _mdf = st.session_state.get("master_df")
    if _mdf is None or _mdf.empty:
        return

    # Build rename map from schema vs current columns
    current_cols = list(_mdf.columns)
    new_cols = [s.get("field", c) for s, c in zip(schema_list, current_cols)]

    # Apply renames if they differ
    rename_map = {}
    for old, new in zip(current_cols, new_cols):
        if old != new and new:
            rename_map[old] = new

    if rename_map:
        _mdf = _mdf.rename(columns=rename_map)
        st.session_state.master_df = _mdf
        print(f"[LiveSync] Schema rename: {rename_map}")

    # Sync schema to DB
    try:
        _active_tbl = st.session_state.get("active_table", "document")
        from utils.sync_engine import SyncEngine
        SyncEngine.sync_schema(schema_list, _active_tbl)
    except Exception as e:
        print(f"[LiveSync] Schema DB sync warning: {e}")

    broadcast_sync(source=source)


def refresh_all_views(source: str = "unknown"):
    """
    Single global refresh: re-derives ALL views from master_json.

    Call this after any direct mutation to master_json (e.g., via set_by_path).
    Sequence:
      1. Re-derive master_df from master_json via ProjectionKernel
      2. Update schema_list from master_json
      3. Broadcast to all aliases (df, csv_df, SQLite)
      4. Fire MASTER_DATA_UPDATED_EVENT
    """
    import copy
    _mj = st.session_state.get("master_json")
    if _mj is None:
        return

    # 1. Re-derive master_df via ProjectionKernel
    _doc_mode = st.session_state.get("doc_mode", "TABLE_MODE")
    try:
        from utils.projection_kernel import ProjectionKernel, PROJECTION_LAYER
        _proj_df, _proj_audit = ProjectionKernel.project(
            _mj, _doc_mode, projection_layer=PROJECTION_LAYER
        )
        if not _proj_df.empty:
            st.session_state.master_df = _proj_df
            st.session_state.df = _proj_df
            st.session_state.csv_df = _proj_df
            print(f"[refresh_all_views] Re-derived master_df ({len(_proj_df)} rows) from master_json")
    except Exception as e:
        print(f"[refresh_all_views] ProjectionKernel error (non-fatal): {e}")

    # 2. Re-derive schema_list
    try:
        from schema_generator import SchemaGenerator
        st.session_state.schema_list = SchemaGenerator.derive_auto(_mj, _doc_mode)
    except Exception as e:
        print(f"[refresh_all_views] Schema derivation error (non-fatal): {e}")

    # 3. Keep json_data alias in sync
    st.session_state.json_data = _mj

    # 4. Broadcast (SQLite sync + fire event)
    broadcast_sync(source=source)
    print(f"[refresh_all_views] Complete (source: {source})")
