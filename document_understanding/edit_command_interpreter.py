"""
Aurora Edit Command Interpreter
=================================
Deterministic NL → JSON patch operation parser.
Zero LLM calls — pure regex + pattern matching.

Converts natural language edit commands into structured JSON operations
following the 5 parsing rules:
  1. Flexible identifier matching (sno50, s.no 50, row 50, #50 → 50)
  2. Command recognition (DELETE/UPDATE/INSERT variants)
  3. Never ask for clarification if inferable
  4. Strict JSON output format
  5. High confidence scoring
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union


# ── Rule 1: Flexible Identifier Patterns ──────────────────────────────────────

_IDENTIFIER_PREFIXES = [
    r"s\.?\s*no\.?",       # sno, s.no, s no, s.no.
    r"sno\.?",             # sno, sno.
    r"serial\s*(?:no\.?)?",  # serial, serial no
    r"row",                # row
    r"record",             # record
    r"entry",              # entry
    r"id",                 # id
    r"item",               # item
    r"line",               # line
    r"#",                  # #50
]

_ID_PATTERN = re.compile(
    r"(?:" + "|".join(_IDENTIFIER_PREFIXES) + r")\s*\.?\s*(\d+)",
    re.IGNORECASE,
)

# Standalone number fallback (e.g., "delete 50")
_BARE_NUMBER = re.compile(r"\b(\d+)\b")


def _extract_row_identifier(text: str) -> Optional[int]:
    """
    Extract row/record identifier from text using flexible matching.
    Rule 1: strip spaces, dots, punctuation between label and number.
    """
    match = _ID_PATTERN.search(text)
    if match:
        return int(match.group(1))
    return None


def _extract_bare_number(text: str, after_keyword: str = "") -> Optional[int]:
    """Extract a standalone number from text, optionally after a keyword."""
    if after_keyword:
        # Find number after the keyword
        pattern = re.compile(
            re.escape(after_keyword) + r"\s+(\d+)", re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    # Fallback: first number in text
    match = _BARE_NUMBER.search(text)
    if match:
        return int(match.group(1))
    return None


# ── Rule 2: Command Recognition ──────────────────────────────────────────────

_DELETE_WORDS = re.compile(
    r"\b(delete|remove|erase|drop|discard|clear|purge)\b", re.IGNORECASE
)

_UPDATE_WORDS = re.compile(
    r"\b(update|edit|change|set|modify|alter|replace|fix|correct|rename)\b",
    re.IGNORECASE,
)

_INSERT_WORDS = re.compile(
    r"\b(add|insert|append|create|new)\b", re.IGNORECASE
)

# Column = value patterns for UPDATE
_UPDATE_COLUMN_VALUE = re.compile(
    r"(\w[\w\s]*?)\s*(?:to|=|:)\s*[\"']?(.+?)[\"']?\s*$",
    re.IGNORECASE,
)

# "set column to value" or "change column to value"
_SET_PATTERN = re.compile(
    r"(?:set|change|update|edit|modify|alter|fix|correct)\s+"
    r"(?:(?:the\s+)?(?:column\s+)?)?(\w[\w\s]*?)\s+"
    r"(?:to|=|as)\s+[\"']?(.+?)[\"']?\s*$",
    re.IGNORECASE,
)

# "update sno 3 name to John" pattern
_UPDATE_ROW_COL_VAL = re.compile(
    r"(?:update|edit|change|set|modify|fix|correct)\s+"
    r"(?:(?:s\.?\s*no\.?|sno\.?|row|record|entry|id|item|#)\s*\.?\s*(\d+))\s+"
    r"(\w[\w\s]*?)\s+"
    r"(?:to|=|as)\s+[\"']?(.+?)[\"']?\s*$",
    re.IGNORECASE,
)

# Key=value pairs for INSERT: "name=Alice, age=30"
_KV_PAIR = re.compile(r"(\w+)\s*=\s*[\"']?([^,\"']+)[\"']?")

# "add new row with ..." pattern
_INSERT_WITH = re.compile(
    r"(?:add|insert|append|create)\s+(?:a\s+)?(?:new\s+)?row\s+(?:with\s+)?(.+)",
    re.IGNORECASE,
)

# "insert row after X" pattern
_INSERT_AFTER = re.compile(
    r"(?:insert|add)\s+(?:a\s+)?(?:new\s+)?row\s+(?:after|before|at)\s+(\d+)",
    re.IGNORECASE,
)

# "rename column X to Y" pattern
_RENAME_PATTERN = re.compile(
    r"rename\s+(?:column\s+)?[\"']?(\w[\w\s]*?)[\"']?\s+(?:to|as)\s+[\"']?(\w[\w\s]*?)[\"']?\s*$",
    re.IGNORECASE,
)


# ── Operation builders ────────────────────────────────────────────────────────


def _build_operation(
    op: str,
    row_identifier: Optional[Union[int, str]] = None,
    column: Optional[str] = None,
    new_value: Any = None,
    reason: str = "",
    confidence: float = 0.95,
    position: Optional[int] = None,
    values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a single operation dict."""
    operation: Dict[str, Any] = {
        "op": op,
        "target": {
            "row_identifier": row_identifier,
            "column": column,
        },
        "new_value": new_value,
        "reason": reason,
        "confidence": round(confidence, 2),
    }
    if position is not None:
        operation["target"]["position"] = position
    if values is not None:
        operation["target"]["values"] = values
    return operation


# ── Main Interpreter ──────────────────────────────────────────────────────────


def interpret_edit_command(command: str) -> Dict[str, Any]:
    """
    Parse a natural language edit command into structured JSON operations.

    Rules applied:
      1. Flexible identifier matching
      2. Command recognition (DELETE/UPDATE/INSERT)
      3. Never ask for clarification if inferable
      4. Strict JSON output format
      5. Confidence scoring

    Args:
        command: Natural language edit command string

    Returns:
        Structured JSON dict with status, operations, and optional clarification
    """
    if not command or not command.strip():
        return {
            "status": "needs_clarification",
            "operations": [],
            "clarification_question": "No command provided. Please specify an action (delete, update, add) and a target.",
        }

    text = command.strip()

    # Try each operation type
    operations: List[Dict[str, Any]] = []

    # ── Check for RENAME ─────────────────────────────────────────────
    rename_match = _RENAME_PATTERN.search(text)
    if rename_match:
        old_name = rename_match.group(1).strip()
        new_name = rename_match.group(2).strip()
        operations.append(_build_operation(
            op="RENAME_COLUMN",
            column=old_name,
            new_value=new_name,
            reason=f"Rename column '{old_name}' to '{new_name}'",
            confidence=0.95,
        ))
        return {
            "status": "ready",
            "operations": operations,
            "clarification_question": None,
        }

    # ── Check for DELETE ─────────────────────────────────────────────
    if _DELETE_WORDS.search(text):
        row_id = _extract_row_identifier(text)
        if row_id is None:
            row_id = _extract_bare_number(text)

        if row_id is not None:
            operations.append(_build_operation(
                op="DELETE_ROW",
                row_identifier=row_id,
                reason=f"Delete row/record #{row_id}",
                confidence=0.95,
            ))
            return {
                "status": "ready",
                "operations": operations,
                "clarification_question": None,
            }

        # Check if deleting a column
        col_match = re.search(
            r"(?:delete|remove|drop)\s+(?:the\s+)?(?:column\s+)?[\"']?(\w[\w\s]*?)[\"']?\s*(?:column)?\s*$",
            text, re.IGNORECASE
        )
        if col_match:
            col_name = col_match.group(1).strip()
            # Filter out row-identifier words
            if col_name.lower() not in ("row", "record", "entry", "item", "line"):
                operations.append(_build_operation(
                    op="DELETE_COLUMN",
                    column=col_name,
                    reason=f"Delete column '{col_name}'",
                    confidence=0.85,
                ))
                return {
                    "status": "ready",
                    "operations": operations,
                    "clarification_question": None,
                }

    # ── Check for UPDATE ─────────────────────────────────────────────
    if _UPDATE_WORDS.search(text):
        # Pattern: "update sno 3 name to John"
        update_match = _UPDATE_ROW_COL_VAL.search(text)
        if update_match:
            row_id = int(update_match.group(1))
            column = update_match.group(2).strip()
            new_value = update_match.group(3).strip()
            new_value = _coerce_value(new_value)

            operations.append(_build_operation(
                op="UPDATE_CELL",
                row_identifier=row_id,
                column=column,
                new_value=new_value,
                reason=f"Update {column} to '{new_value}' in row #{row_id}",
                confidence=0.95,
            ))
            return {
                "status": "ready",
                "operations": operations,
                "clarification_question": None,
            }

        # Pattern: "change row 3 price to 500"
        row_id = _extract_row_identifier(text)
        if row_id is not None:
            # Remove identifier part and action word to get column=value
            cleaned = _ID_PATTERN.sub("", text)
            cleaned = _UPDATE_WORDS.sub("", cleaned).strip()

            col_val_match = _UPDATE_COLUMN_VALUE.search(cleaned)
            if col_val_match:
                column = col_val_match.group(1).strip()
                new_value = _coerce_value(col_val_match.group(2).strip())

                operations.append(_build_operation(
                    op="UPDATE_CELL",
                    row_identifier=row_id,
                    column=column,
                    new_value=new_value,
                    reason=f"Update {column} to '{new_value}' in row #{row_id}",
                    confidence=0.90,
                ))
                return {
                    "status": "ready",
                    "operations": operations,
                    "clarification_question": None,
                }

        # Pattern: "set/change column to value" (no row specified — global)
        set_match = _SET_PATTERN.search(text)
        if set_match:
            column = set_match.group(1).strip()
            new_value = _coerce_value(set_match.group(2).strip())
            operations.append(_build_operation(
                op="UPDATE_CELL",
                row_identifier=None,
                column=column,
                new_value=new_value,
                reason=f"Set {column} to '{new_value}'",
                confidence=0.75,
            ))
            return {
                "status": "ready",
                "operations": operations,
                "clarification_question": None,
            }

    # ── Check for INSERT ─────────────────────────────────────────────
    if _INSERT_WORDS.search(text):
        # Pattern: "insert row after 5"
        after_match = _INSERT_AFTER.search(text)
        if after_match:
            position = int(after_match.group(1))
            operations.append(_build_operation(
                op="INSERT_ROW",
                position=position,
                reason=f"Insert new row after position {position}",
                confidence=0.90,
            ))
            return {
                "status": "ready",
                "operations": operations,
                "clarification_question": None,
            }

        # Pattern: "add new row with name=Alice, age=30"
        with_match = _INSERT_WITH.search(text)
        if with_match:
            kv_text = with_match.group(1)
            pairs = _KV_PAIR.findall(kv_text)
            if pairs:
                values = {}
                for k, v in pairs:
                    values[k.strip()] = _coerce_value(v.strip())

                operations.append(_build_operation(
                    op="INSERT_ROW",
                    values=values,
                    reason=f"Add new row with {len(values)} field(s)",
                    confidence=0.90,
                ))
                return {
                    "status": "ready",
                    "operations": operations,
                    "clarification_question": None,
                }

        # Generic "add row"
        operations.append(_build_operation(
            op="INSERT_ROW",
            reason="Insert new empty row",
            confidence=0.70,
        ))
        return {
            "status": "ready",
            "operations": operations,
            "clarification_question": None,
        }

    # ── Rule 3: Try harder before asking for clarification ────────────
    # Check if there's any identifiable number + action word combo
    row_id = _extract_row_identifier(text) or _extract_bare_number(text)
    has_action = (_DELETE_WORDS.search(text) or _UPDATE_WORDS.search(text)
                  or _INSERT_WORDS.search(text))

    if row_id and not has_action:
        # Has identifier but no clear action
        return {
            "status": "needs_clarification",
            "operations": [],
            "clarification_question": (
                f"I found row/record #{row_id}, but I'm not sure what action to perform. "
                "Did you mean to delete, update, or insert?"
            ),
        }

    # ── No recognizable pattern → needs clarification ────────────────
    return {
        "status": "needs_clarification",
        "operations": [],
        "clarification_question": (
            "I couldn't identify a clear action (delete/update/add) or target. "
            "Please use commands like: 'delete sno 5', 'update row 3 name to John', "
            "or 'add new row with name=Alice'."
        ),
    }


def _coerce_value(value: str) -> Any:
    """Coerce a string value to int/float/bool if possible."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none", "nil"):
        return None

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    return value


# ── Batch interpreter ─────────────────────────────────────────────────────────


def interpret_batch_commands(commands: List[str]) -> Dict[str, Any]:
    """
    Interpret multiple edit commands and merge into a single response.
    """
    all_operations = []
    clarifications = []

    for cmd in commands:
        result = interpret_edit_command(cmd)
        if result["status"] == "ready":
            all_operations.extend(result["operations"])
        elif result.get("clarification_question"):
            clarifications.append(f"[{cmd}]: {result['clarification_question']}")

    if clarifications and not all_operations:
        return {
            "status": "needs_clarification",
            "operations": [],
            "clarification_question": " | ".join(clarifications),
        }

    return {
        "status": "ready" if all_operations else "needs_clarification",
        "operations": all_operations,
        "clarification_question": " | ".join(clarifications) if clarifications else None,
    }


# ── Pretty printer ────────────────────────────────────────────────────────────


def format_result(result: Dict[str, Any]) -> str:
    """Pretty-print the interpreter result."""
    return json.dumps(result, indent=2, default=str)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_commands = [
        "delete sno50",
        "delete sno 50",
        "remove row 50",
        "erase entry 50",
        "delete s.no 50",
        "update sno 3 name to John",
        "change row 3 price to 500",
        "set name to Alice",
        "add new row with name=Alice, age=30, city=Dubai",
        "insert row after 5",
        'edit sno. 7 total to 1500.75',
        "rename column Tax ID to TaxNo",
        "remove #42",
        "drop record 99",
        "",
        "hello",
        "row 50",
    ]

    print("=" * 70)
    print("Aurora Edit Command Interpreter — Test Suite")
    print("=" * 70)

    for cmd in test_commands:
        print(f"\nInput: \"{cmd}\"")
        result = interpret_edit_command(cmd)
        print(format_result(result))
