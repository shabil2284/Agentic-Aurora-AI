"""
Aurora Path Utilities — utils/path_utils.py
============================================
Path-based get/set for master_json binding.

Supports dot and bracket notation:
  - "Items[0].Amount"
  - "topLevel"
  - "meta.values[2].name"

Used by:
  - Tabular Viewer  (write-back via set_by_path)
  - Undo system     (inverse ops reference paths)
  - Chatbot Editor  (context anchoring)

NEVER modifies extraction, schema generation, or pipeline logic.
"""

import copy
import re
from typing import Any, Dict, List, Optional, Tuple


def _tokenize(path: str) -> List[str]:
    """Split a dot/bracket path into tokens."""
    return [t for t in path.replace("]", "").split("[") if t]


def _expand(raw_tokens: List[str]) -> List[str]:
    """Expand 'a.b' tokens into ['a', 'b']."""
    out = []
    for t in raw_tokens:
        out.extend(t.split("."))
    return [t for t in out if t]


def get_by_path(obj: Any, path: str) -> Any:
    """
    Read a value from a nested dict/list using dot/bracket path.

    >>> get_by_path({"Items": [{"Amount": 100}]}, "Items[0].Amount")
    100
    """
    if not path:
        return None
    tokens = _expand(_tokenize(path))
    cur = obj
    for t in tokens:
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(t)]
            except (IndexError, ValueError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(t)
        else:
            return None
    return cur


def set_by_path(obj: Any, path: str, value: Any) -> bool:
    """
    Write a value into a nested dict/list at the given path.
    Creates intermediate dicts/lists if missing.

    >>> d = {"Items": [{"Amount": 100}]}
    >>> set_by_path(d, "Items[0].Amount", 200)
    True
    >>> d["Items"][0]["Amount"]
    200
    """
    if not path:
        return False
    tokens = _expand(_tokenize(path))
    cur = obj
    for i, t in enumerate(tokens):
        if i == len(tokens) - 1:
            # Final token — set value
            if isinstance(cur, list):
                try:
                    cur[int(t)] = value
                    return True
                except (IndexError, ValueError):
                    return False
            elif isinstance(cur, dict):
                cur[t] = value
                return True
            return False

        # Navigate deeper
        next_t = tokens[i + 1]
        is_next_index = next_t.isdigit()

        if isinstance(cur, list):
            try:
                idx = int(t)
                if cur[idx] is None:
                    cur[idx] = [] if is_next_index else {}
                cur = cur[idx]
            except (IndexError, ValueError):
                return False
        elif isinstance(cur, dict):
            if t not in cur or cur[t] is None:
                cur[t] = [] if is_next_index else {}
            cur = cur[t]
        else:
            return False
    return False


def build_binding_map(
    master_json: dict,
    array_key: str,
) -> List[Dict[str, str]]:
    """
    For a hierarchical master_json with an entity-list array,
    build binding paths for each row/cell.

    Returns a list of dicts: [{col_name: "Items[i].col_name", ...}, ...]
    """
    arr = master_json.get(array_key, [])
    if not isinstance(arr, list):
        return []

    bindings = []
    for idx, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        row_bindings = {}
        for col in item.keys():
            row_bindings[col] = f"{array_key}[{idx}].{col}"
        bindings.append(row_bindings)
    return bindings


def find_array_key(master_json: dict) -> Optional[str]:
    """
    Find the best entity-list array key in a hierarchical master_json.
    Returns the key name or None.
    """
    if not isinstance(master_json, dict):
        return None

    best_key = None
    best_count = 0
    for k, v in master_json.items():
        if isinstance(v, list) and len(v) > 0:
            n_dicts = sum(1 for x in v if isinstance(x, dict))
            if n_dicts >= len(v) * 0.5 and n_dicts > best_count:
                best_key = k
                best_count = n_dicts
    return best_key


def resolve_field_paths(
    obj: Any,
    field_name: str,
    current_path: Optional[List] = None,
) -> List[List]:
    """
    Recursively scan the entire master_json and return ALL paths
    where `field_name` appears as a key.

    Returns a list of paths, where each path is a list of keys/indices.

    Example:
      obj = {"Companies": [{"Email": "a@b.com"}, {"Email": "c@d.com"}], "Email": "root@x.com"}
      resolve_field_paths(obj, "Email")
      -> [["Email"], ["Companies", 0, "Email"], ["Companies", 1, "Email"]]
    """
    if current_path is None:
        current_path = []
    results = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = current_path + [k]
            if k.lower() == field_name.lower():
                results.append(new_path)
            # Recurse into nested structures
            if isinstance(v, (dict, list)):
                results.extend(resolve_field_paths(v, field_name, new_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            new_path = current_path + [idx]
            if isinstance(item, (dict, list)):
                results.extend(resolve_field_paths(item, field_name, new_path))

    return results


def set_value_at_path(obj: Any, path: List, value: Any) -> bool:
    """
    Set a value in a nested dict/list using a path list.

    >>> d = {"Items": [{"Amount": 100}]}
    >>> set_value_at_path(d, ["Items", 0, "Amount"], 999)
    True
    >>> d["Items"][0]["Amount"]
    999
    """
    if not path:
        return False
    cur = obj
    for key in path[:-1]:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return False
    try:
        cur[path[-1]] = value
        return True
    except (KeyError, IndexError, TypeError):
        return False


def get_value_at_path(obj: Any, path: List) -> Any:
    """
    Get a value from a nested dict/list using a path list.
    """
    cur = obj
    for key in path:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return None
    return cur
