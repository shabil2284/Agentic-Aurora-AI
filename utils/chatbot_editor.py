"""
Aurora Chatbot Editor — utils/chatbot_editor.py
================================================
Translates natural language to structured JSON responses:
  - type: "answer"   — analytics/read queries (totals, counts, lookups)
  - type: "proposal" — staged edit operations (DELETE/UPDATE/INSERT/RENAME)
  - type: "clarify"  — ambiguous requests requiring user disambiguation

NON-INTERFERENCE: Does NOT modify extraction, table creation, SQL engine, or
export generation logic. Pure NL→JSON translation layer.
"""
import json
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils.groq_client import get_groq_client


class EditCondition(BaseModel):
    column: str = Field(..., description="The exact column name to check")
    operator: str = Field(..., description="E.g., '=', '!=', '>', '<', 'contains'")
    value: Any = Field(..., description="The value to check against")


class AtomicOperation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = Field(
        ...,
        description="RENAME_COLUMN, CHANGE_TYPE, UPDATE_CELL, DELETE_ROW, ADD_ROW, ADD_COLUMN, DELETE_COLUMN",
    )
    target: Dict[str, Any] = Field(
        default_factory=dict,
        description="Target metadata, e.g. {'old': 'Tax ID', 'new': 'TaxNo'} or {'col_name': 'NewColumn'}.",
    )
    condition: Optional[EditCondition] = Field(
        None,
        description="Explicit search condition for DELETE_ROW, UPDATE_CELL.",
    )
    before: Optional[Any] = Field(None, description="Value before the change (for audit)")
    after: Optional[Any] = Field(None, description="Value after the change (for audit)")
    reason: str = Field(..., description="Brief explanation of why this edit is proposed.")
    confidence: float = Field(default=1.0)


class EditProposal(BaseModel):
    operations: List[AtomicOperation]
    clarification_needed: Optional[str] = Field(
        None,
        description="If the request is ambiguous, ask a clarifying question here.",
    )
    summary: Optional[str] = Field(None, description="One-line summary of proposed changes")


# ── Translator System Prompt ─────────────────────────────────────────────────
TRANSLATOR_SYSTEM_PROMPT = """You are Document Edit Translator. Input includes: master_json, table_schema, table_rows_sample, user_query.
Output MUST be JSON only. One of:
- {{"type":"answer","text":"..."}}
- {{"type":"clarify","questions":["..."]}}
- {{"type":"proposal","operations":[ ... ],"summary":"..."}}

Operation schema:
{{ "op_id": "uuid", "op":"DELETE_ROW"|"UPDATE_CELL"|"INSERT_ROW"|"RENAME_COLUMN", "target": {{"table":"...","where":{{<col>:<val>}}, "row_index":<int>}}, "column":"Name", "before":<value>, "after":<value>, "confidence":0.0 }}

Rules:
- Always resolve ambiguous column names to canonical columns (include sample values for each candidate).
- If ambiguous, output type:clarify with targeted questions (show sample values).
- For destructive operations affecting >1 row, require typed confirmation before Apply (client will handle).
- For analytics queries (total sales, profit, count, etc.) return type:answer with the numeric result and minimal explanation; also include the SQL query you would run (SQL: "...") for auditability.
- Do not ever produce plain text outside the JSON object."""


def generate_atomic_edits(
    user_request: str,
    schema_context: Dict[str, Any],
    sample_data: List[Dict[str, Any]],
) -> EditProposal:
    """
    Translates a natural language user request into a strict list of AtomicOperations.
    Requires schema context and sample row data to accurately map fuzzy column names.
    """
    client = get_groq_client()
    llm = client.with_structured_output(EditProposal)

    system_prompt = f"""
You are the Aurora God-Level Chatbot Editor Engine.
Your job is to translate a user's natural language edit request into a strict JSON array of AtomicOperations.

AVAILABLE OPERATION TYPES (You MUST use conditions for row edits):
- RENAME_COLUMN: target must have 'old' (exact current column name) and 'new' (new column name).
- CHANGE_TYPE: target must have 'col' (column name) and 'new_type' (e.g., numeric, string, date).
- DELETE_ROW: MUST have `condition` (e.g., column="amount", operator="=", value=48).
- UPDATE_CELL: MUST have `condition` (to find the row) and `target` (e.g. {{'col': 'Total', 'new_val': 100}}).
- ADD_ROW: target must have 'data' dict matching the schema.
- ADD_COLUMN: target must have 'col_name' and 'default_val'.
- DELETE_COLUMN: target must have 'col_name' in target.
- UPDATE_JSON_FIELD: For nested/hierarchical fields. target must have 'field_name' (field key to find) and 'new_val'. Optionally 'json_path' (list of keys/indices).
  Use this when editing fields that are NOT flat table columns (e.g., "Invoice Total", "Bank details.CEO", "Payment Details").

CURRENT SCHEMA Context (Top-Level Keys or DataFrame Columns):
{json.dumps(schema_context, indent=2)}

SAMPLE DATA Context:
{json.dumps(sample_data[:2], indent=2)}

CRITICAL RULES:
1. You must resolve fuzzy names to EXACT schema names. If the user says "Delete tax id", and the schema has "Tax ID", your target MUST use "Tax ID".
2. For DELETE_ROW or UPDATE_CELL, ALWAYS populate the `condition` object with {{"column": "ExactName", "operator": "=", "value": ...}}. Supported operators: '=', '!=', '>', '<', 'contains'.
3. If the request is highly ambiguous (e.g., user says "Delete it" without specifying what), set `clarification_needed` with a question.
4. Keep the `reason` field short.
"""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        result_dict = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_request),
        ])
        return EditProposal(**result_dict) if isinstance(result_dict, dict) else result_dict
    except Exception as e:
        print(f"[ChatbotEditor] Failed to parse NL->Atomic JSON. Error: {e}")
        return EditProposal(
            operations=[],
            clarification_needed="I had trouble generating the exact change instructions. Could you rephrase your edit request?",
        )


def generate_analytics_answer(
    user_query: str,
    schema_context: Dict[str, Any],
    data_context: str,
    schema_lines: str = "",
) -> Dict[str, Any]:
    """
    Handles read-only analytics queries. Returns structured response:
    {"type": "answer", "text": "..."} or {"type": "clarify", "questions": [...]}
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    system_prompt = f"""You are Aurora, an enterprise AI document assistant.
You have access to the live extracted document data and the current schema below.
IMPORTANT RULES:
- Answer ONLY from the data provided below.
- The schema shows the current field names; if a field was renamed or deleted, it is reflected here.
- If a field no longer exists in the schema or data, say so clearly.
- Do NOT reference field names that don't appear in the Current Schema.
- If the data doesn't contain the answer, say so briefly and honestly.
- For numeric answers, show the computation and result clearly.
- For aggregate queries (total, sum, average, count), compute from the data and present concisely.

{schema_lines}

EXTRACTED DATA:
{data_context}
"""

    try:
        llm = get_groq_client("llama-3.1-8b-instant")
        print("[Chatbot] Using model: llama-3.1-8b-instant")
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ])
        return {"type": "answer", "text": response.content}
    except Exception as e:
        print(f"[Chatbot] Primary model failed: {e}, trying fallback...")
        try:
            llm = get_groq_client("gemma2-9b-it")
            print("[Chatbot] Using model: gemma2-9b-it (fallback)")
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query),
            ])
            return {"type": "answer", "text": response.content}
        except Exception as e2:
            print(f"[Chatbot] Fallback also failed: {e2}")
            return {"type": "answer", "text": f"I encountered an error processing your query."}
