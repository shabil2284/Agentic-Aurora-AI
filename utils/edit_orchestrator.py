"""
Aurora Edit Orchestrator — utils/edit_orchestrator.py
=====================================================
Executes staged AtomicOperations against master_df, generates inverse_ops
for full undo capability, and triggers live sync.

NON-INTERFERENCE: Does NOT modify extraction, table creation, or SQL engine.
Only mutates master_df and delegates persistence to update_master_model.
"""
import copy
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Any


def apply_staged_changes(
    staged_ops: List[Dict],
    master_json: Any,
    master_df: pd.DataFrame,
) -> Tuple[bool, str, List[Dict]]:
    """
    Executes JSON atomic operations against master_df.
    Returns (success, message, inverse_ops).
    inverse_ops is populated on success so that undo can reverse the commit.
    """
    if master_df is None or master_df.empty:
        return False, "Active data table is empty. Cannot process edits.", []

    df = master_df.copy()
    inverse_ops: List[Dict] = []

    try:
        for op in staged_ops:
            op_type = op.get("type", "")
            target = op.get("target", {})

            if op_type == "RENAME_COLUMN":
                old_name = target.get("old")
                new_name = target.get("new")
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
                    inverse_ops.append({
                        "type": "RENAME_COLUMN",
                        "target": {"old": new_name, "new": old_name},
                    })
                else:
                    return False, f"Column '{old_name}' not found for RENAME_COLUMN.", []

            elif op_type == "DELETE_COLUMN":
                col_name = target.get("col_name")
                if col_name in df.columns:
                    # Cache column data for inverse
                    _col_data = df[col_name].tolist()
                    _col_idx = list(df.columns).index(col_name)
                    df = df.drop(columns=[col_name])
                    inverse_ops.append({
                        "type": "RESTORE_COLUMN",
                        "target": {"col_name": col_name, "data": _col_data, "position": _col_idx},
                    })
                else:
                    return False, f"Column '{col_name}' not found for DELETE_COLUMN.", []

            elif op_type == "ADD_COLUMN":
                col_name = target.get("col_name")
                default_val = target.get("default_val", None)
                if col_name not in df.columns:
                    df[col_name] = default_val
                    inverse_ops.append({
                        "type": "DELETE_COLUMN",
                        "target": {"col_name": col_name},
                    })
                else:
                    return False, f"Column '{col_name}' already exists.", []

            elif op_type == "RESTORE_COLUMN":
                # Inverse of DELETE_COLUMN — used during undo
                col_name = target.get("col_name")
                col_data = target.get("data", [])
                col_pos = target.get("position", len(df.columns))
                if col_name not in df.columns:
                    df.insert(col_pos, col_name, col_data[:len(df)] if col_data else None)
                    inverse_ops.append({
                        "type": "DELETE_COLUMN",
                        "target": {"col_name": col_name},
                    })

            elif op_type in ["DELETE_ROW", "UPDATE_CELL"]:
                condition = op.get("condition", {})
                if not condition:
                    return False, f"Missing required 'condition' for {op_type}.", []

                col = condition.get("column")
                operator = condition.get("operator")
                val = condition.get("value")

                if col not in df.columns:
                    return False, f"Condition column '{col}' not found in data.", []

                # Build pandas boolean mask
                try:
                    if operator in ("=", "=="):
                        mask = df[col].astype(str) == str(val)
                    elif operator == "!=":
                        mask = df[col].astype(str) != str(val)
                    elif operator == ">":
                        mask = df[col] > float(val)
                    elif operator == "<":
                        mask = df[col] < float(val)
                    elif operator == "contains":
                        mask = df[col].astype(str).str.contains(str(val), case=False, na=False)
                    else:
                        return False, f"Unsupported condition operator: {operator}", []
                except Exception as eval_e:
                    return False, f"Failed to evaluate condition '{col} {operator} {val}': {str(eval_e)}", []

                if op_type == "DELETE_ROW":
                    # Cache deleted rows for inverse (INSERT_ROW)
                    deleted_rows = df[mask].to_dict(orient="records")
                    deleted_indices = df[mask].index.tolist()
                    df = df[~mask].reset_index(drop=True)
                    # Inverse: re-insert each deleted row
                    for row_idx, row_data in zip(reversed(deleted_indices), reversed(deleted_rows)):
                        inverse_ops.append({
                            "type": "INSERT_ROW",
                            "target": {"data": row_data, "index": row_idx},
                        })

                elif op_type == "UPDATE_CELL":
                    new_val = target.get("new_val")
                    tgt_col = target.get("col")
                    if tgt_col not in df.columns:
                        return False, f"Target column '{tgt_col}' not found for UPDATE_CELL.", []
                    # Cache old values for inverse
                    old_values = df.loc[mask, tgt_col].tolist()
                    old_indices = df[mask].index.tolist()
                    df.loc[mask, tgt_col] = new_val
                    # Inverse: restore each cell to its old value
                    for idx, old_val in zip(old_indices, old_values):
                        inverse_ops.append({
                            "type": "UPDATE_CELL",
                            "condition": condition,
                            "target": {"col": tgt_col, "new_val": old_val},
                            "_restore_index": idx,
                        })

            elif op_type == "INSERT_ROW":
                row_data = target.get("data", {})
                insert_idx = target.get("index")
                if insert_idx is not None and insert_idx <= len(df):
                    # Insert at specific position
                    new_row = pd.DataFrame([row_data])
                    df = pd.concat([df.iloc[:insert_idx], new_row, df.iloc[insert_idx:]]).reset_index(drop=True)
                else:
                    new_row = pd.DataFrame([row_data])
                    df = pd.concat([df, new_row]).reset_index(drop=True)
                # Inverse: delete the inserted row
                inverse_ops.append({
                    "type": "DELETE_ROW",
                    "condition": {"column": list(row_data.keys())[0], "operator": "=", "value": list(row_data.values())[0]} if row_data else {},
                    "target": {},
                })

            elif op_type == "ADD_ROW":
                row_data = target.get("data", {})
                new_row = pd.DataFrame([row_data])
                df = pd.concat([df, new_row]).reset_index(drop=True)
                inverse_ops.append({
                    "type": "DELETE_ROW",
                    "condition": {"column": list(row_data.keys())[0], "operator": "=", "value": list(row_data.values())[0]} if row_data else {},
                    "target": {},
                })

            elif op_type == "UPDATE_JSON_FIELD":
                # Direct master_json path-based edit (for nested/hierarchical fields)
                field_name = target.get("field_name", "")
                new_val = target.get("new_val")
                json_path = target.get("json_path")  # optional explicit path

                if not field_name and not json_path:
                    return False, "UPDATE_JSON_FIELD requires 'field_name' or 'json_path' in target.", []

                from utils.path_utils import resolve_field_paths, set_value_at_path, get_value_at_path

                _mj = st.session_state.get("master_json")
                if _mj is None:
                    return False, "master_json is not available.", []

                _updated_json = copy.deepcopy(_mj)

                if json_path:
                    # Explicit path provided
                    old_val = get_value_at_path(_updated_json, json_path)
                    if set_value_at_path(_updated_json, json_path, new_val):
                        inverse_ops.append({
                            "type": "UPDATE_JSON_FIELD",
                            "target": {"field_name": field_name, "new_val": old_val, "json_path": json_path},
                        })
                        print(f"[EditOrch] UPDATE_JSON_FIELD via explicit path: {json_path}")
                    else:
                        return False, f"Failed to set value at path {json_path}.", []
                else:
                    # Resolve ALL matching paths
                    paths = resolve_field_paths(_updated_json, field_name)
                    if not paths:
                        return False, f"Field '{field_name}' not found anywhere in master_json.", []

                    for p in paths:
                        old_val = get_value_at_path(_updated_json, p)
                        set_value_at_path(_updated_json, p, new_val)
                        inverse_ops.append({
                            "type": "UPDATE_JSON_FIELD",
                            "target": {"field_name": field_name, "new_val": old_val, "json_path": p},
                        })
                    print(f"[EditOrch] UPDATE_JSON_FIELD '{field_name}': updated {len(paths)} paths")

                # Commit master_json and refresh all views
                st.session_state.master_json = _updated_json
                st.session_state.json_data = _updated_json
                from utils.live_sync import refresh_all_views
                refresh_all_views(source="UPDATE_JSON_FIELD")

                # Skip the df-based commit below — already handled
                inverse_ops.reverse()
                return True, f"JSON field '{field_name}' updated at {len(paths) if not json_path else 1} location(s) and all views refreshed.", inverse_ops

        # ── Commit: update session state and trigger live sync ──
        st.session_state.master_df = df

        # Also sync changes back to master_json via path resolution
        _mj = st.session_state.get("master_json")
        if _mj is not None:
            from utils.path_utils import resolve_field_paths, set_value_at_path

            _updated_json = copy.deepcopy(_mj)
            _json_changed = False

            for op in staged_ops:
                op_type = op.get("type", "")

                # ── RENAME_COLUMN: rename keys in master_json ──
                if op_type == "RENAME_COLUMN":
                    _tgt = op.get("target", {})
                    _old_name = _tgt.get("old")
                    _new_name = _tgt.get("new")
                    if _old_name and _new_name:
                        paths = resolve_field_paths(_updated_json, _old_name)
                        for p in paths:
                            # Navigate to parent dict, pop old key, set new key
                            parent = _updated_json
                            for key in p[:-1]:
                                try:
                                    parent = parent[key]
                                except (KeyError, IndexError, TypeError):
                                    parent = None
                                    break
                            if isinstance(parent, dict) and _old_name in parent:
                                parent[_new_name] = parent.pop(_old_name)
                                _json_changed = True
                                print(f"[EditOrch] Synced RENAME_COLUMN '{_old_name}' → '{_new_name}' in master_json")

                # ── DELETE_COLUMN: remove keys from master_json ──
                elif op_type == "DELETE_COLUMN":
                    _tgt = op.get("target", {})
                    _col_name = _tgt.get("col_name")
                    if _col_name:
                        paths = resolve_field_paths(_updated_json, _col_name)
                        for p in reversed(paths):  # reverse to not invalidate indices
                            parent = _updated_json
                            for key in p[:-1]:
                                try:
                                    parent = parent[key]
                                except (KeyError, IndexError, TypeError):
                                    parent = None
                                    break
                            if isinstance(parent, dict) and _col_name in parent:
                                del parent[_col_name]
                                _json_changed = True
                                print(f"[EditOrch] Synced DELETE_COLUMN '{_col_name}' in master_json")

                # ── ADD_COLUMN: add keys to record dicts in master_json ──
                elif op_type == "ADD_COLUMN":
                    _tgt = op.get("target", {})
                    _col_name = _tgt.get("col_name")
                    _default_val = _tgt.get("default_val", None)
                    if _col_name:
                        # Add the key to every record-level dict
                        def _add_key_to_records(obj):
                            changed = False
                            if isinstance(obj, dict):
                                # Check if this looks like a data record (has multiple scalar fields)
                                scalars = sum(1 for v in obj.values() if not isinstance(v, (dict, list)))
                                if scalars >= 2 and _col_name not in obj:
                                    obj[_col_name] = _default_val
                                    changed = True
                                for v in obj.values():
                                    if _add_key_to_records(v):
                                        changed = True
                            elif isinstance(obj, list):
                                for item in obj:
                                    if _add_key_to_records(item):
                                        changed = True
                            return changed
                        if _add_key_to_records(_updated_json):
                            _json_changed = True
                            print(f"[EditOrch] Synced ADD_COLUMN '{_col_name}' in master_json")

                # ── UPDATE_CELL: update values in master_json (existing logic) ──
                elif op_type == "UPDATE_CELL":
                    _tgt = op.get("target", {})
                    _col = _tgt.get("col")
                    _new_val = _tgt.get("new_val")
                    if _col:
                        paths = resolve_field_paths(_updated_json, _col)
                        # Apply to matching array paths (condition-based filtering)
                        _cond = op.get("condition", {})
                        _cond_col = _cond.get("column")
                        _cond_val = str(_cond.get("value", ""))

                        for p in paths:
                            # Verify the condition matches this specific path
                            if _cond_col and len(p) >= 2:
                                # Get the parent object of this field
                                parent_path = p[:-1]
                                parent_obj = _updated_json
                                for k in parent_path:
                                    try:
                                        parent_obj = parent_obj[k]
                                    except (KeyError, IndexError, TypeError):
                                        parent_obj = None
                                        break
                                if isinstance(parent_obj, dict) and str(parent_obj.get(_cond_col, "")) == _cond_val:
                                    set_value_at_path(_updated_json, p, _new_val)
                                    _json_changed = True
                                    print(f"[EditOrch] Synced UPDATE_CELL to master_json path: {p}")
                            else:
                                set_value_at_path(_updated_json, p, _new_val)
                                _json_changed = True

            if _json_changed:
                st.session_state.master_json = _updated_json
                st.session_state.json_data = _updated_json

        from utils.live_sync import update_master_model
        update_master_model(df=df, source="CHATBOT_EDITOR")

        # Reverse inverse_ops so undo applies in correct order
        inverse_ops.reverse()

        return True, "Data successfully mutated and synchronized.", inverse_ops

    except Exception as e:
        return False, f"Exception during operation execution: {str(e)}", []


def undo_staged_changes(
    inverse_ops: List[Dict],
    master_json: Any,
    master_df: pd.DataFrame,
) -> Tuple[bool, str]:
    """
    Applies pre-computed inverse_ops to reverse a committed change.
    Uses apply_staged_changes internally for consistency.
    """
    if not inverse_ops:
        return False, "No inverse operations available for undo."

    success, msg, _ = apply_staged_changes(inverse_ops, master_json, master_df)
    return success, msg
