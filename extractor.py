import json
import re
from typing import Any, Dict, List, Optional
from utils.strict_llm_client import StrictLLMClient

# =================================================================
# AURORA v2 -- MODE-SPECIFIC EXTRACTION PROMPTS
# =================================================================

# -- Mode 1: TABLE_MODE -------------------------------------------
TABLE_MODE_PROMPT = """
AURORA v2 - TABLE MODE EXTRACTION

YOU ARE: Azure Form Recognizer + AWS Textract + Google Document AI.

CRITICAL RULE: ZERO HALLUCINATION POLICY
Extract ONLY values literally present in the document.
Do NOT invent, guess, or fabricate any value.

DOCUMENT TYPE: List / Directory / Table / Spreadsheet

LAYOUT MODE - ONE LINE = ONE RECORD:
  Each line is exactly one record. Never merge or split lines.

STRICT NUMBERED-LIST PATTERN:
  [SNo]. [Company Name] [separator] [Email]
  [SNo]. [Company Name] [separator] [Email] [separator] [Phone]
  Separators recognized: -  |  /  tab

  SNo          -> number before the dot
  Company Name -> ALL words from after the dot to the FIRST separator (never truncate)
  Email        -> segment containing "@"
  Phone        -> segment matching digit/+/- phone pattern
  Website      -> segment with www/.com but NOT "@"

CRITICAL - DO NOT BREAK COMPOUND NAMES:
  "Michael Page UAE"     -> Company Name: Michael Page UAE   CORRECT
  "Robert Half UAE"      -> Company Name: Robert Half UAE    CORRECT
  "Charterhouse Middle East" -> Company Name: Charterhouse Middle East  CORRECT
  NEVER: Company Name: Michael  WRONG
  NEVER: Company Name: Page UAE  WRONG

FIELD TYPE DETECTION:
  "@" in value            -> Email  (ALWAYS - IRON RULE)
  digits + +/-/spaces 7+  -> Phone
  www/.com/http (no "@")  -> Website
  pure number at start    -> SNo
  street/road/city/nagar  -> Address

MARKDOWN LINKS:
  [text](mailto:email@x.com) -> email@x.com
  [text](https://x.com)      -> https://x.com

NULL HANDLING:
  If field missing -> skip the line entirely.
  NEVER output: Record X Field: NULL / N/A / None

COMPLETENESS: Extract EVERY row. If 100 rows -> 100 records.

OUTPUT FORMAT (non-negotiable):
  Record X FieldName: Value
  (one field per line, no JSON, no CSV, no markdown, no comments)

Example:
  Record 26 SNo: 26
  Record 26 Company Name: TASC Outsourcing
  Record 26 Email: careers@tascoutsourcing.com
  Record 27 SNo: 27
  Record 27 Company Name: Hays UAE
  Record 27 Email: dubai@hays.com
  Record 28 SNo: 28
  Record 28 Company Name: Michael Page UAE
  Record 28 Email: michaelpage@ae.michaelpage.com

{SCHEMA_INJECTION}
"""

# -- Mode 2: HIERARCHICAL_MODE ------------------------------------
HIERARCHICAL_MODE_PROMPT = """
AURORA v2 - HIERARCHICAL MODE EXTRACTION

YOU ARE: Azure Form Recognizer Layout + AWS Textract AnalyzeDocument.

CRITICAL RULE: ZERO HALLUCINATION - output only what exists in the document.

DOCUMENT TYPE: Resume / CV / Invoice / Contract / Report / Form with Sections

LAYOUT UNDERSTANDING:
  Headings         -> top-level keys in the JSON object
  Bullet points    -> elements inside an array
  Sub-sections     -> nested objects inside arrays
  Key: Value pairs -> direct key-value in the parent object

NEVER flatten hierarchical data into rows.
NEVER collapse an Experience section into a single string.

EXTRACTION RULES:
1. Identify all section headings (Summary, Experience, Education, Projects, Skills, etc.)
2. For each section:
   - Simple text -> string value
   - List items -> JSON array of strings or objects
   - Subsections (job entries, education entries) -> array of objects
3. For job entries, extract: Company, Role, Date, Description (array of bullet strings)
4. For education: Institution, Degree, Date, Grade
5. For projects: Name, Description, Tech stack
6. For skills: preserve as array of strings

NULL HANDLING:
  If a field has no value in the document -> omit it entirely from output.
  Never output null, N/A, or placeholder values.

OUTPUT FORMAT: Single valid JSON object ONLY.
  Do NOT output Record X lines.
  Do NOT output CSV.
  Do NOT add any text before or after the JSON.

Example for Resume:
{{
  "Name": "John Smith",
  "Email": "john@email.com",
  "Phone": "+1 555 0100",
  "Summary": "Senior software engineer with 8 years experience...",
  "Experience": [
    {{
      "Company": "Acme Corp",
      "Role": "Senior Engineer",
      "Date": "2020 - Present",
      "Description": ["Led backend architecture", "Reduced latency by 40%"]
    }}
  ],
  "Education": [
    {{
      "Institution": "MIT",
      "Degree": "B.Sc. Computer Science",
      "Date": "2015 - 2019"
    }}
  ],
  "Skills": ["Python", "Kubernetes", "PostgreSQL"],
  "Projects": [
    {{
      "Name": "OpenPay",
      "Description": "Payment gateway",
      "Tech": ["Node.js", "Stripe API"]
    }}
  ]
}}

Example for Invoice:
{{
  "Invoice No": "INV-2024-001",
  "Date": "2024-01-15",
  "Vendor": "Acme Supplies Ltd",
  "Client": "Gulf Trading LLC",
  "Line Items": [
    {{"Description": "Web Server", "Qty": "2", "Unit Price": "$500", "Total": "$1000"}}
  ],
  "Subtotal": "$1000",
  "Tax": "$50",
  "Grand Total": "$1050"
}}

{SCHEMA_INJECTION}
"""

# -- Mode 3: KEY_VALUE_MODE ---------------------------------------
KEY_VALUE_MODE_PROMPT = """
AURORA v2 - KEY-VALUE MODE EXTRACTION

YOU ARE: Azure Form Recognizer + Google Document AI.

CRITICAL RULE: ZERO HALLUCINATION - extract only what is visible.

DOCUMENT TYPE: Application Form / ID Card / Registration Form / Simple Form

EXTRACTION RULES:
1. Find every field label -> value pair in the document
2. Extract the label exactly as printed
3. Extract the filled/written value exactly as it appears
4. If a field has no value written -> omit it

NULL HANDLING:
  Blank fields -> omit from output entirely.
  NEVER output: "Field: N/A" or "Field: null"

OUTPUT FORMAT: Single valid JSON object.
  No Record X lines. No CSV. No extra text.

Example:
{{
  "Full Name": "Jane Doe",
  "Date of Birth": "12/05/1990",
  "Nationality": "UAE",
  "Passport No": "P1234567",
  "Email": "jane@email.com",
  "Phone": "+971 50 123 4567"
}}

{SCHEMA_INJECTION}
"""

# -- Mode Detection Prompt ----------------------------------------
MODE_DETECTION_PROMPT = """You are a document layout classifier. Analyze the document sample below and respond with EXACTLY ONE of:
  TABLE_MODE
  HIERARCHICAL_MODE
  KEY_VALUE_MODE

Rules:
- TABLE_MODE: numbered lists, directories, recruitment lists, multi-row tables, CSV-like structure
- HIERARCHICAL_MODE: resume/CV with sections (Summary/Experience/Education/Skills), invoices with line items, contracts with clauses
- KEY_VALUE_MODE: simple forms, ID cards, application fields with labels and filled values

Respond with ONLY the mode name. No explanation. No punctuation.

DOCUMENT SAMPLE:
{sample}
"""


def _clean_value(v: Any) -> str:
    """Remove markdown email/URL wrappers from extracted values."""
    if not isinstance(v, str):
        return v
    # [text](mailto:email) -> email
    m = re.match(r'\[([^\]]*)\]\(mailto:([^)]+)\)', v.strip(), re.IGNORECASE)
    if m:
        return m.group(2).strip()
    # [text](https://url) -> url
    m2 = re.match(r'\[([^\]]*)\]\((https?://[^)]+)\)', v.strip(), re.IGNORECASE)
    if m2:
        return m2.group(2).strip()
    return v


class Extractor:
    VALID_MODES = {"TABLE_MODE", "HIERARCHICAL_MODE", "KEY_VALUE_MODE"}

    def __init__(self):
        self.llm = StrictLLMClient(require_json=False)

    # -- Step 1: Mode Detection -----------------------------------
    def detect_mode(self, text: str) -> str:
        """Fast pre-pass: classify document into TABLE / HIERARCHICAL / KEY_VALUE mode."""
        from langchain_core.messages import SystemMessage, HumanMessage
        sample = text[:2000]
        prompt = MODE_DETECTION_PROMPT.replace("{sample}", sample)
        messages = [
            SystemMessage(content="You are a document layout classifier. Reply with ONLY the mode name."),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages, {}, max_retries=1)
        raw = response.get("data", "") if response.get("status") in ["success", "repaired"] else ""
        detected = raw.strip().upper().split()[0] if raw.strip() else "TABLE_MODE"
        if detected not in self.VALID_MODES:
            if "HIER" in detected or "RESUME" in detected or "CV" in detected:
                detected = "HIERARCHICAL_MODE"
            elif "KEY" in detected or "FORM" in detected:
                detected = "KEY_VALUE_MODE"
            else:
                detected = "TABLE_MODE"
        print(f"[Aurora v2] Document mode detected: {detected}")
        return detected

    # -- Step 2: Extract data with mode-specific prompt -----------
    def extract_data(self, text: str, schema: Dict[str, Any] = None, errors: List[str] = None, **kwargs) -> Dict[str, Any]:
        from langchain_core.messages import SystemMessage, HumanMessage

        orch_mode = kwargs.get("mode", "EXTRACT_KV")
        target_tb = kwargs.get("target_table", "")
        execute_perm = "true" if kwargs.get("execute_permission") else "false"
        cloud_prov = kwargs.get("cloud_provider", "")

        schema_text = (
            "USE THIS SPECIFIC SCHEMA:\n\n" + json.dumps(schema, indent=2)
            if schema else
            "DYNAMICALLY DETECT SCHEMA FROM DOCUMENT LAYOUT. DO NOT SKIP NESTED FIELDS."
        )

        # Mode detection pre-pass
        doc_mode = self.detect_mode(text)

        # Select prompt and output strategy
        if doc_mode == "HIERARCHICAL_MODE":
            prompt_template = HIERARCHICAL_MODE_PROMPT
            user_instruction = "RESPOND WITH A SINGLE VALID JSON OBJECT ONLY. NO RECORD LINES. NO EXTRA TEXT."
        elif doc_mode == "KEY_VALUE_MODE":
            prompt_template = KEY_VALUE_MODE_PROMPT
            user_instruction = "RESPOND WITH A SINGLE VALID JSON OBJECT ONLY. NO EXTRA TEXT."
        else:
            doc_mode = "TABLE_MODE"
            prompt_template = TABLE_MODE_PROMPT
            user_instruction = "RESPOND WITH ONLY 'Record X FieldName: Value' LINES. NO JSON. NO CSV."

        prompt_str = prompt_template.replace("{SCHEMA_INJECTION}", schema_text)
        prompt_str += f"""
---
ORCHESTRATOR:
mode: {orch_mode}  |  table: {target_tb}  |  execute: {execute_perm}  |  cloud: {cloud_prov}
DOC_MODE: {doc_mode}
"""
        CHUNK_SIZE = 15000
        MAX_CHUNKS = 30 if doc_mode == "TABLE_MODE" else 10
        text_chunks = [text[i:i+CHUNK_SIZE] for i in range(0, min(len(text), CHUNK_SIZE * MAX_CHUNKS), CHUNK_SIZE)]

        import concurrent.futures
        import time

        def process_chunk(chunk):
            messages = [
                SystemMessage(content=prompt_str),
                HumanMessage(content=f"DOCUMENT CONTENT:\n```text\n{chunk}\n```\n\n{user_instruction}")
            ]
            for attempt in range(4):
                response = self.llm.invoke(messages, {}, max_retries=1)
                if response.get("status") in ["success", "repaired"]:
                    return response
                err = str(response.get("last_error", ""))
                if any(x in err for x in ["429", "413", "Quota", "exhausted", "too large"]):
                    time.sleep(20 * (attempt + 1))
                else:
                    return response
            return response

        # Collect results in SUBMISSION ORDER (not completion order)
        # as_completed() returns in arrival order which shuffles rows between chunks
        chunk_responses = [None] * len(text_chunks)
        total_chunks = len(text_chunks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_idx = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(text_chunks)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunk_responses[idx] = future.result()
                except Exception as exc:
                    if errors is not None:
                        errors.append(f"Chunk exception: {exc}")
                    chunk_responses[idx] = {"status": "failed", "last_error": str(exc)}
                completed = sum(1 for r in chunk_responses if r is not None)
                print(f"   [Aurora v2] {doc_mode} Chunk {completed}/{total_chunks} done", flush=True)
        # Remove any None slots from failed futures
        chunk_responses = [r for r in chunk_responses if r is not None]

        # Parse output based on mode
        if doc_mode == "TABLE_MODE":
            final_data = self._parse_table_output(chunk_responses, errors)
        else:
            final_data = self._parse_json_output(chunk_responses, errors)

        # ── Fallback 1: Entity List Detection for KEY_VALUE_MODE ────────────
        # (Fix for when LLM only returned a title instead of the repeating unlabelled entities)
        _all_raw_text = "\n".join(
            r.get("data", "") if isinstance(r.get("data"), str) else ""
            for r in chunk_responses
            if r.get("status") in ["success", "repaired"]
        )
        if doc_mode == "KEY_VALUE_MODE":
            _is_list, _list_output = self._parse_implicit_pairs(_all_raw_text, text, check_entity_list_only=True)
            if _is_list:
                print(f"[Aurora v2] Strict Entity List pattern detected. Activating ENTITY_LIST_MODE.")
                final_data = _list_output
                doc_mode = "ENTITY_LIST_MODE"

        # ── Fallback 2: if LLM returned nothing, try general implicit pair detection ──────
        _is_empty = (
            (isinstance(final_data, list) and len(final_data) == 0) or
            (isinstance(final_data, dict) and len(final_data) == 0)
        )
        if _is_empty and doc_mode != "ENTITY_LIST_MODE":
            print("[Aurora v2] LLM returned empty — trying implicit pair fallback")
            _, _fallback = self._parse_implicit_pairs(_all_raw_text, text, check_entity_list_only=False)
            if _fallback:
                print(f"[Aurora v2] Implicit pair fallback extracted {len(_fallback)} records")
                final_data = _fallback

        return {"data": final_data, "doc_mode": doc_mode}

    # -- TABLE_MODE parser: "Record X Field: Value" lines --------
    def _parse_table_output(self, chunk_responses, errors):
        records_map = {}
        global_record = {}
        for chunk_idx, response in enumerate(chunk_responses):  # chunk_idx now always in submission order
            if response.get("status") not in ["success", "repaired"]:
                if errors:
                    errors.append(f"Chunk failed: {response.get('last_error')}")
                continue
            raw = response.get("data", "")
            if not isinstance(raw, str):
                continue
            for line in raw.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith(("{" , "}", "[", "]", '"')):
                    continue
                if ":" not in line:
                    continue
                try:
                    key, value = line.split(":", 1)
                    k, v = key.strip(), value.strip()
                    if not k or not v:
                        continue
                    if k.startswith('"') or v.startswith(('{', '[')):
                        continue
                    m = re.match(r'(?i)^Record\s+(\S+)\s+(.*)', k)
                    if m:
                        # rec_id = chunk_order + record_number to prevent cross-chunk collision
                        # while still keeping chunk 0 records before chunk 1 records
                        rec_id = f"{chunk_idx:06d}_{m.group(1)}"
                        real_k = m.group(2).strip()
                        if not real_k:
                            continue
                        records_map.setdefault(rec_id, {})[real_k] = _clean_value(v)
                    else:
                        global_record[k] = _clean_value(v)
                except ValueError:
                    pass
        if records_map:
            # Sort by rec_id to guarantee chunk order is preserved in output
            return [records_map[k] for k in sorted(records_map.keys())]
        return [global_record] if global_record else []

    # -- HIERARCHICAL / KEY_VALUE parser: JSON output -------------
    def _parse_json_output(self, chunk_responses, errors):
        """Merge JSON objects from all chunks into one unified structure."""
        merged = {}
        for response in chunk_responses:
            if response.get("status") not in ["success", "repaired"]:
                if errors:
                    errors.append(f"Chunk failed: {response.get('last_error')}")
                continue
            raw = response.get("data", "")
            if isinstance(raw, dict):
                obj = raw
            elif isinstance(raw, str):
                clean = raw.strip()
                clean = re.sub(r'^```json\s*', '', clean, flags=re.IGNORECASE)
                clean = re.sub(r'```\s*$', '', clean)
                s = clean.find('{')
                e = clean.rfind('}')
                if s != -1 and e != -1:
                    try:
                        obj = json.loads(clean[s:e+1])
                    except Exception:
                        continue
                else:
                    continue
            else:
                continue

            # Merge: arrays accumulate, scalars take first non-empty value
            for k, v in obj.items():
                if k not in merged:
                    merged[k] = v
                elif isinstance(merged[k], list) and isinstance(v, list):
                    merged[k].extend(v)
                elif not merged[k] and v:
                    merged[k] = v

        return merged if merged else {}

    # -- FALLBACK: implicit pair pattern detection ----------------
    def _parse_implicit_pairs(self, llm_text: str, original_text: str, check_entity_list_only: bool = False) -> tuple:
        """
        Fallback for documents that have no explicit keys — just alternating lines
        such as:
            Acme Corp
            https://maps.app.goo.gl/abc123

        Returns: (is_entity_list_mode, resulting_data)
        """
        source = original_text.strip() if original_text.strip() else llm_text.strip()
        if not source:
            return False, []

        _maps_re   = re.compile(r'https?://maps\.app\.goo\.gl\S+', re.IGNORECASE)
        _url_re    = re.compile(r'https?://\S+', re.IGNORECASE)
        _phone_re  = re.compile(r'[\+\d][\d\s\-\.\(\)]{6,}\d')

        def _classify(line: str):
            if _maps_re.search(line):
                return "Google Maps", _maps_re.search(line).group(0)
            if _url_re.search(line):
                return "Website", _url_re.search(line).group(0)
            if _phone_re.search(line) and not any(c.isalpha() for c in line[:5]):
                return "Phone", line.strip()
            return "Company Name", line.strip()

        lines = [l.strip() for l in source.splitlines() if l.strip() and len(l.strip()) > 2]

        # Detect alternating pattern: check both even and odd indices for URLs
        odd_urls = sum(1 for i, l in enumerate(lines) if i % 2 == 1 and _url_re.search(l))
        even_urls = sum(1 for i, l in enumerate(lines) if i > 0 and i % 2 == 0 and _url_re.search(l))
        
        pair_mode = False
        start_idx = 0
        
        if len(lines) >= 4:
            if odd_urls / max(len(lines) // 2, 1) > 0.4:
                pair_mode = True
                start_idx = 0
            elif even_urls / max(len(lines) // 2, 1) > 0.4:
                pair_mode = True
                start_idx = 1

        if check_entity_list_only and not pair_mode:
            return False, []

        records = []
        doc_title = "Extracted Entity List"
        
        if pair_mode:
            # Use lines before start_idx as potential title
            if start_idx > 0:
                doc_title = " ".join(lines[:start_idx]).strip()

            for i in range(start_idx, len(lines) - 1, 2):
                label_a, val_a = _classify(lines[i])
                label_b, val_b = _classify(lines[i + 1])
                record = {}
                if label_a != "Company Name":
                    label_a, label_b = label_b, label_a
                    val_a,   val_b   = val_b,   val_a
                record[label_a] = val_a
                record[label_b] = val_b
                records.append(record)
                
            if check_entity_list_only:
                # Format exactly as requested for ENTITY_LIST_MODE
                return True, {
                    "Document Title": doc_title,
                    "Companies": records
                }
            return True, records
            
        else:
            current: dict = {}
            for line in lines:
                label, val = _classify(line)
                if label == "Company Name":
                    if current:
                        records.append(current)
                    current = {"Company Name": val}
                else:
                    if not current:
                        current = {}
                    existing = current.get(label)
                    if existing:
                        current[f"{label} 2"] = val
                    else:
                        current[label] = val
            if current:
                records.append(current)

        return False, (records if records else [])

