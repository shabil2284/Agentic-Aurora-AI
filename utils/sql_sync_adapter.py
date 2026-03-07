"""
Aurora SQL Sync Adapter — utils/sql_sync_adapter.py
=====================================================
Handles table creation, transactional upsert with optimistic locking,
and conflict detection for the tabularization module.
"""

import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple
from utils.sync_engine import SyncEngine

# Column name for optimistic locking
ROW_VERSION_COL = "__row_version"
SOURCE_ROW_COL  = "__source_row_id"

# Meta columns that should NOT appear in the DB as data columns
META_COLS = {"__master_path", "__created_at_confidence"}


def _infer_sql_type(value: Any) -> str:
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _clean_col_name(name: str) -> str:
    """Sanitise column names for SQLite."""
    import re
    return re.sub(r"[^\w]", "_", name).strip("_") or "col"


class SQLSyncAdapter:
    """
    Creates and syncs Aurora tabular data to an SQLite table.
    Uses optimistic locking via __row_version to detect conflicts.
    """

    def __init__(self):
        self.db_path = SyncEngine.get_db_path()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Schema Inference ───────────────────────────────────────────────────────

    def infer_schema(self, rows: List[dict]) -> Dict[str, str]:
        """
        Return {col_name: sql_type} dict for all non-meta columns.
        Uses votes across first 20 rows to pick the best type.
        """
        schema: Dict[str, str] = {}
        for row in rows[:20]:
            for k, v in row.items():
                if k in META_COLS:
                    continue
                col = _clean_col_name(k)
                t = _infer_sql_type(v)
                # TEXT is the safe default; prefer specific if already set to TEXT
                if schema.get(col, "TEXT") == "TEXT" and t != "TEXT":
                    schema[col] = t
                elif col not in schema:
                    schema[col] = t
        # Ensure PK and version columns exist
        schema[_clean_col_name(SOURCE_ROW_COL)] = "TEXT"
        schema[ROW_VERSION_COL] = "INTEGER"
        return schema

    # ── Table Creation (Step 1) ────────────────────────────────────────────────

    def create_table_schema(self, tabular_preview: dict, table_name: str) -> bool:
        """
        Creates the SQL table with inferred schema. Safe — does NOT insert rows.
        Returns True on success.
        """
        rows = tabular_preview.get("rows", [])
        if not rows:
            print("[SQLSyncAdapter] No rows in preview — skipping table creation.")
            return False

        schema = self.infer_schema(rows)
        pk_col = _clean_col_name(SOURCE_ROW_COL)

        col_defs = []
        for col, typ in schema.items():
            if col == pk_col:
                col_defs.append(f'"{col}" {typ} PRIMARY KEY')
            else:
                col_defs.append(f'"{col}" {typ}')

        ddl = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(col_defs)})'

        conn = self._connection()
        try:
            with conn:
                conn.execute(ddl)
            print(f"[SQLSyncAdapter] Table '{table_name}' created/verified.")
            return True
        except Exception as e:
            print(f"[SQLSyncAdapter] Table creation failed: {e}")
            return False
        finally:
            conn.close()

    # ── Conflict Detection ────────────────────────────────────────────────────

    def detect_conflicts(self, staged_rows: List[dict], table_name: str) -> List[dict]:
        """
        Returns list of conflict dicts where staged __row_version != DB __row_version.
        """
        if not SyncEngine.table_exists(table_name):
            return []

        pk_col = _clean_col_name(SOURCE_ROW_COL)
        conn = self._connection()
        conflicts = []
        try:
            for row in staged_rows:
                rid = row.get(SOURCE_ROW_COL)
                staged_version = row.get(ROW_VERSION_COL, 0)
                cur = conn.execute(
                    f'SELECT "{ROW_VERSION_COL}" FROM "{table_name}" WHERE "{pk_col}" = ?',
                    (rid,)
                )
                db_row = cur.fetchone()
                if db_row and db_row[ROW_VERSION_COL] != staged_version:
                    conflicts.append({
                        "__source_row_id": rid,
                        "staged_version": staged_version,
                        "db_version": db_row[ROW_VERSION_COL],
                    })
        except Exception as e:
            print(f"[SQLSyncAdapter] Conflict detection error: {e}")
        finally:
            conn.close()
        return conflicts

    # ── Transactional Upsert (Step 2 AutoSync) ────────────────────────────────

    def upsert_rows(self, rows: List[dict], table_name: str) -> Tuple[int, List[dict]]:
        """
        Upserts all rows into the table using optimistic locking.
        Returns (upserted_count, conflicts_list).
        All writes happen inside one transaction; rolls back on error.
        """
        if not SyncEngine.table_exists(table_name):
            print(f"[SQLSyncAdapter] Table '{table_name}' does not exist. Run create_table_schema first.")
            return 0, []

        conflicts = self.detect_conflicts(rows, table_name)
        if conflicts:
            print(f"[SQLSyncAdapter] {len(conflicts)} conflicts detected — upsert aborted.")
            return 0, conflicts

        pk_col = _clean_col_name(SOURCE_ROW_COL)
        conn = self._connection()
        upserted = 0
        try:
            conn.execute("BEGIN")
            for row in rows:
                # Build clean row (no meta columns except PK + version)
                db_row: Dict[str, Any] = {}
                for k, v in row.items():
                    if k in META_COLS:
                        continue
                    db_row[_clean_col_name(k)] = v
                # Bump version
                db_row[ROW_VERSION_COL] = int(db_row.get(ROW_VERSION_COL, 0)) + 1

                cols   = list(db_row.keys())
                values = list(db_row.values())
                ph     = ", ".join(["?"] * len(cols))
                update_clause = ", ".join(
                    [f'"{c}" = excluded."{c}"' for c in cols if c != pk_col]
                )
                sql = (
                    f'INSERT INTO "{table_name}" ({", ".join(f"{chr(34)}{c}{chr(34)}" for c in cols)}) '
                    f'VALUES ({ph}) '
                    f'ON CONFLICT("{pk_col}") DO UPDATE SET {update_clause}'
                )
                conn.execute(sql, values)
                upserted += 1

            conn.execute("COMMIT")
            print(f"[SQLSyncAdapter] Upserted {upserted} rows → '{table_name}'.")
        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"[SQLSyncAdapter] Upsert FAILED, rolled back: {e}")
            return 0, [{"error": str(e)}]
        finally:
            conn.close()

        return upserted, []
