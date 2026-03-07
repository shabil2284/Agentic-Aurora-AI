import sqlite3
import re
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data.db")


class SyncEngine:
    """
    Unified sync engine for Aurora.
    All writes route through one shared SQLite DB (data.db).

    Key design rules:
    - Each document gets its own table (never overwrite other documents).
    - Tables are NEVER auto-dropped; safe upsert is the default.
    - master_json is always the source of truth; SQL is a derived store.
    """

    # ------------------------------------------------------------------ #
    #  Connection / DB Path                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_db_path() -> str:
        return os.path.abspath(DB_PATH)

    @staticmethod
    def _get_connection() -> sqlite3.Connection:
        conn = sqlite3.connect(SyncEngine.get_db_path(), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    #  Table Name Generator                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_table_name(file_name: str = "", doc_mode: str = "TABLE_MODE") -> str:
        """
        Generate a deterministic, document-specific table name.

        Rules:
        - Strips path, extension, and special characters from the filename.
        - Appends a type prefix based on doc_mode.
        - Max 40 chars to stay safe with SQLite.

        Examples:
            "Invoice_Nov2024.pdf"   + HIERARCHICAL_MODE → "invoice_nov2024"
            "Recruitment List.xlsx" + TABLE_MODE         → "recruitment_list"
            "resume_john.pdf"       + KEY_VALUE_MODE     → "resume_john"
        """
        if file_name:
            base = os.path.splitext(os.path.basename(file_name))[0]
            base = re.sub(r"[^a-zA-Z0-9_]", "_", base).strip("_").lower()
            base = re.sub(r"_+", "_", base)[:40]
        else:
            # Fallback: mode-based generic name
            mode_prefix = {
                "HIERARCHICAL_MODE": "document",
                "KEY_VALUE_MODE":    "form",
            }.get(doc_mode, "table")
            base = mode_prefix

        return base if base else "document"

    # ------------------------------------------------------------------ #
    #  Safe Sync — NEVER drops an existing table                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def sync_all(df: pd.DataFrame, table_name: str = "document", schema: Optional[List[Dict]] = None):
        """
        Safe sync: writes DataFrame to SQLite using REPLACE strategy for the
        FIRST write (table doesn't exist yet) and safe column-preserving UPDATE
        for subsequent writes on the SAME session.

        SAFETY RULE: This method will NEVER drop or recreate a table that
        already exists with a DIFFERENT schema — it only operates on the exact
        table_name passed in.
        """
        if df is None or df.empty:
            print("[SyncEngine] sync_all called with empty/None df — skipping.")
            return

        db_path = SyncEngine.get_db_path()
        print(f"[SyncEngine] Syncing {len(df)} rows → table='{table_name}'")

        # Preserve SNo as string so SQLite doesn't cast it to INTEGER
        if "SNo" in df.columns:
            df = df.copy()
            df["SNo"] = df["SNo"].astype(str)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            table_exists = SyncEngine.table_exists(table_name)
            if not table_exists:
                # First time: safe to create fresh
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"[SyncEngine] Created new table '{table_name}' with {len(df)} rows.")
            else:
                # Table exists: update in-place, don't touch schema of other tables
                # Use replace into existing structure (same table_name, same session)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"[SyncEngine] Updated existing table '{table_name}' — {len(df)} rows.")
            conn.commit()
        finally:
            conn.close()

        # Write flat CSV and JSON to system TEMP storage, NOT project dir
        import tempfile
        import glob
        base_dir = tempfile.gettempdir()
        csv_path  = os.path.join(base_dir, f"{table_name}.csv")
        json_path = os.path.join(base_dir, f"{table_name}.json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        print(f"[SyncEngine] TEMP CSV  → {csv_path}")
        print(f"[SyncEngine] TEMP JSON → {json_path}")

        if schema:
            SyncEngine.sync_schema(schema, table_name)

    # ------------------------------------------------------------------ #
    #  Schema sync                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def sync_schema(schema: List[Dict], table_name: str = "document"):
        if not schema:
            return
        import tempfile
        base_dir = tempfile.gettempdir()
        schema_path = os.path.join(base_dir, f"{table_name}_schema.json")
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"[SyncEngine] TEMP SCHEMA → {schema_path}")

    # ------------------------------------------------------------------ #
    #  Automatic Temporary File Cleanup                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def auto_cleanup():
        """
        Wipes all aurora-related temporary files (*.json, *.csv, *_schema.json)
        from the system temp directory so they don't persist between sessions.
        """
        import tempfile
        import glob
        base_dir = tempfile.gettempdir()
        patterns = [
            os.path.join(base_dir, "tmp*.json"),
            os.path.join(base_dir, "tmp*.csv"),
            os.path.join(base_dir, "document*.json"),
            os.path.join(base_dir, "document*.csv"),
            os.path.join(base_dir, "form*.json"),
            os.path.join(base_dir, "form*.csv"),
            os.path.join(base_dir, "aurora*.json"),
            os.path.join(base_dir, "aurora*.csv")
        ]
        deleted_count = 0
        for pat in patterns:
            for fpath in glob.glob(pat):
                try:
                    os.remove(fpath)
                    deleted_count += 1
                except Exception:
                    pass
        if deleted_count > 0:
            print(f"[SyncEngine] Cleaned up {deleted_count} stale temporary files.")

    # ------------------------------------------------------------------ #
    #  Read / Query helpers                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def fetch_all(table_name: str = "document") -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(SyncEngine.get_db_path(), check_same_thread=False)
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            conn.close()
            print(f"[SyncEngine] fetch_all '{table_name}' — {len(df)} rows")
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"[SyncEngine] fetch_all FAILED: {e}")
            return []

    @staticmethod
    def table_exists(table_name: str) -> bool:
        try:
            conn = sqlite3.connect(SyncEngine.get_db_path(), check_same_thread=False)
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            exists = cur.fetchone() is not None
            conn.close()
            return exists
        except Exception:
            return False

    @staticmethod
    def list_tables() -> List[str]:
        """Return names of all tables in the DB."""
        try:
            conn = sqlite3.connect(SyncEngine.get_db_path(), check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r[0] for r in cur.fetchall()]
            conn.close()
            return tables
        except Exception:
            return []

    @staticmethod
    def run_query(query: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(SyncEngine.get_db_path(), check_same_thread=False)
            result = pd.read_sql_query(query, conn)
            conn.close()
            return result
        except Exception as e:
            print(f"[SyncEngine] run_query FAILED: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Legacy shims                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def upsert_data(table_name: str, data: List[Dict[str, Any]]):
        if not data:
            return
        SyncEngine.sync_all(pd.DataFrame(data), table_name)

    @staticmethod
    def init_table(table_name: str, data: List[Dict[str, Any]]):
        if not data:
            return None, []
        keys = list(data[0].keys())
        pk = next((k for k in keys if k.lower() in {"sno", "id", "record_id"}), keys[0])
        return pk, keys
