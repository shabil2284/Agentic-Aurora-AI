import sqlite3
import re
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from utils.sync_engine import SyncEngine

DB_PATH = SyncEngine.get_db_path()


class SQLEngine:
    def __init__(self, db_url: Optional[str] = None):
        self.db_path = DB_PATH
        print(f"[SQLEngine] Connected to: {self.db_path}")

    # ------------------------------------------------------------------ #
    #  Connection                                                           #
    # ------------------------------------------------------------------ #
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    #  Live Schema                                                          #
    # ------------------------------------------------------------------ #
    def get_tables(self) -> List[str]:
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            conn.close()
            return tables
        except Exception:
            return []

    def get_columns(self, table: str) -> List[str]:
        """Return the exact column names for a table from SQLite PRAGMA."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in cur.fetchall()]
            conn.close()
            return cols
        except Exception:
            return []

    def get_schema_info(self, table_name: str = "users") -> str:
        """Return formatted schema string listing all tables and their columns."""
        lines = []
        for tbl in self.get_tables():
            cols = self.get_columns(tbl)
            lines.append(f"TABLE {tbl}:\n  " + "\n  ".join(cols) if cols else f"TABLE {tbl}: (empty)")
        return "\n".join(lines) if lines else "No tables found."

    def get_active_table(self, hint: str = "") -> str:
        """Return the best active table name — prefer the hint, fall back to first available."""
        tables = self.get_tables()
        if not tables:
            return "users"
        if hint and hint in tables:
            return hint
        # Prefer 'users' as the canonical table name if it exists
        if "users" in tables:
            return "users"
        return tables[0]

    # ------------------------------------------------------------------ #
    #  Natural Language -> SQL                                             #
    # ------------------------------------------------------------------ #
    def _generate_column_aliases(self, columns: List[str]) -> Dict[str, str]:
        """Automatically create a semantic alias map for actual table columns."""
        aliases = {}
        for col in columns:
            col_lower = col.lower()
            aliases[col_lower] = col
            # Common semantic permutations
            if "invoice" in col_lower:
                aliases["invoice"] = col
                aliases["bill"] = col
            if "vendor" in col_lower:
                aliases["vendor"] = col
                aliases["company"] = col
                aliases["merchant"] = col
                aliases["seller"] = col
            if "client" in col_lower or "customer" in col_lower or "buyer" in col_lower:
                aliases["client"] = col
                aliases["customer"] = col
                aliases["buyer"] = col
            if "amount" in col_lower or "total" in col_lower or "price" in col_lower:
                aliases["amount"] = col
                aliases["total"] = col
                aliases["price"] = col
            if "date" in col_lower:
                aliases["date"] = col
        return aliases

    def _resolve_values(self, natural_query: str, active_table: str, live_columns: List[str]) -> str:
        """
        Scan the actual SQLite DB to find exactly which column contains
        the numbers/words mentioned in the natural language query.
        Returns a strict hint string for the LLM.
        """
        import re
        # Extract potential values (numbers, consecutive chars)
        tokens = re.findall(r'\b[a-zA-Z0-9.\-@]+\b', natural_query)
        # Filter out common SQL/English stop words
        stopwords = {"delete", "remove", "update", "change", "set", "show", "get", "find", "all", "where", "to", "from", "row", "rows", "invoice", "vendor", "client", "amount", "total", "date", "table", "data", "is", "by", "in", "and", "or"}
        search_terms = [t for t in tokens if t.lower() not in stopwords and len(t) > 1]
        
        if not search_terms:
            return ""

        hints = []
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            for term in search_terms:
                matched_cols = []
                for col in live_columns:
                    # Search if the term exists in this column anywhere in the table
                    safe_term = term.replace("'", "''")
                    cur.execute(f"SELECT 1 FROM \"{active_table}\" WHERE \"{col}\" LIKE '%{safe_term}%' LIMIT 1")
                    if cur.fetchone():
                        matched_cols.append(col)
                if len(matched_cols) == 1:
                    hints.append(f"Value '{term}' STRICTLY exists only in the \"{matched_cols[0]}\" column.")
                elif len(matched_cols) > 1:
                    hints.append(f"Value '{term}' exists in multiple columns: {', '.join(matched_cols)}.")
            conn.close()
        except Exception as e:
            print(f"[SQLEngine] Value resolution failed: {e}")
            
        if hints:
            return "DATABASE VALUE SCAN RESULTS:\n" + "\n".join(f"  - {h}" for h in hints)
        return ""

    def generate_sql(self, natural_query: str, schema_info: Any = None, table_hint: str = "") -> str:
        """
        Generate strict, schema-grounded SQL from natural language.
        Uses ONLY columns that actually exist in the DB.
        Enforces safe DELETE semantics.
        """
        from utils.groq_client import get_groq_client
        from langchain_core.messages import SystemMessage, HumanMessage

        active_table = self.get_active_table(table_hint)
        live_columns = self.get_columns(active_table)
        live_schema  = self.get_schema_info()

        # Strict Resolution Injection
        col_aliases = self._generate_column_aliases(live_columns)
        alias_str = "\n".join(f"  - '{k}' refers to \"{v}\"" for k, v in col_aliases.items())
        value_hints = self._resolve_values(natural_query, active_table, live_columns)

        col_list = "\n".join(f"  - \"{c}\"" for c in live_columns)

        system_prompt = f"""You are Aurora SQL Intelligence Engine.

ACTIVE TABLE: "{active_table}"

EXACT COLUMNS (use ONLY these — never invent):
{col_list}

SEMANTIC COLUMN ALIASES:
{alias_str}

{value_hints}

FULL DATABASE SCHEMA:
{live_schema}

STRICT RULES:
1. ALWAYS quote table names and column names with double quotes: "{active_table}", "Column Name".
2. Use ONLY column names listed above — never invent new ones. If a Semantic Alias matches, use its EXACT COLUMN mapping.
3. If Database Value Scan Results are provided, you MUST use the column it says the value is inside of.
4. DELETE SAFETY & LOGIC:
   - "delete [column]" → UPDATE "{active_table}" SET "[column]" = NULL
   - "delete row where [condition]" → DELETE FROM "{active_table}" WHERE [condition]
   - "delete [alias] [value]" → Example: "delete invoice 5049" → DELETE FROM "{active_table}" WHERE "Invoice No" = '5049'
5. If user says "show", "get", "find", "fetch" → generate SELECT.
6. If user says "change [alias] to [value] where [condition]" → generate UPDATE "{active_table}" SET ... WHERE ...
7. String values must be wrapped in single quotes.
8. LIKE '%value%' for partial string matches.
9. Return ONLY the raw SQL. No markdown. No explanation. No prefix.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate SQL for: {natural_query}")
        ]

        # Use the highly rigid 8b model first to prevent hallucination overhangs from 70b
        for model in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"]:
            try:
                llm = get_groq_client(model)
                response = llm.invoke(messages)
                sql_clean = response.content.strip()
                # Strip markdown code fences
                sql_clean = re.sub(r'^```sql\s*', '', sql_clean, flags=re.IGNORECASE)
                sql_clean = re.sub(r'^```\s*', '', sql_clean)
                sql_clean = re.sub(r'```\s*$', '', sql_clean).strip()
                print(f"[SQLEngine] Generated SQL ({model}): {sql_clean}")
                return sql_clean
            except Exception as e:
                print(f"[SQLEngine] Model {model} failed: {e}")
                continue

        return "SELECT 1; -- All models failed"

    # ------------------------------------------------------------------ #
    #  Column Validation                                                    #
    # ------------------------------------------------------------------ #
    def validate_sql(self, sql: str, active_table: str) -> Tuple[bool, str]:
        """
        Pre-flight check: verify that all columns referenced in the SQL
        actually exist in the active table schema.
        Returns (ok, error_message).
        """
        known_cols = set(self.get_columns(active_table))
        if not known_cols:
            return True, ""  # Can't validate if table doesn't exist yet

        sql_upper = sql.upper()
        # Extract tokens that could be column names (simple heuristic)
        # Only warn — don't block, to avoid false positives on SQLite keywords
        tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', sql)
        sql_keywords = {
            "SELECT", "FROM", "WHERE", "UPDATE", "SET", "DELETE", "INSERT",
            "INTO", "VALUES", "AND", "OR", "NOT", "NULL", "IS", "LIKE",
            "ORDER", "BY", "LIMIT", "GROUP", "HAVING", "AS", "JOIN",
            "LEFT", "RIGHT", "INNER", "OUTER", "ON", "DISTINCT", "ALL",
            "IN", "BETWEEN", "CASE", "WHEN", "THEN", "ELSE", "END",
            "TABLE", "ALTER", "ADD", "COLUMN", "TEXT", "INTEGER", "REAL",
            str(active_table).upper()
        }
        suspicious = [
            t for t in tokens
            if t.upper() not in sql_keywords
            and t not in known_cols
            and len(t) > 2
        ]
        if suspicious:
            print(f"[SQLEngine] Unrecognized tokens in SQL: {suspicious}")
            # Not a hard block — LLM might have aliased or used functions
        return True, ""

    # ------------------------------------------------------------------ #
    #  Execution                                                            #
    # ------------------------------------------------------------------ #
    def execute_raw(self, sql: str) -> Tuple[pd.DataFrame, str]:
        """Execute SQL against the shared SQLite DB. Returns (result_df, message)."""
        print(f"[SQLEngine] execute_raw: {sql}")
        try:
            conn = self._get_connection()
            sql_stripped = sql.strip().upper()
            if sql_stripped.startswith("SELECT") or sql_stripped.startswith("WITH"):
                df = pd.read_sql_query(sql, conn)
                conn.close()
                return df, f"Query returned {len(df)} rows."
            else:
                cur = conn.cursor()
                cur.execute(sql)
                conn.commit()
                rowcount = cur.rowcount
                conn.close()
                return pd.DataFrame(), f"Executed successfully. Rows affected: {rowcount}"
        except Exception as e:
            print(f"[SQLEngine] execute_raw FAILED: {e}")
            return pd.DataFrame(), f"Error: {str(e)}"

    def save_dataframe(self, df: pd.DataFrame, table_name: str = "users"):
        SyncEngine.sync_all(df, table_name)
