import re

with open("extractor.py", "r") as f:
    content = f.read()

new_prompt = '''ANTI-GRAVITY PROJECT: STRUCTURED KEY VALUE EXTRACTION WITH CSV EDITOR AND JSON AUTOSYNC

SYSTEM ROLE:

You are Anti-Gravity Structured Extraction and Editor Engine.

PRIMARY OBJECTIVE:

Extract 100% Key Value pairs from document.

Then render:

LEFT PANEL → Editable CSV Table (Scrollable)
RIGHT PANEL → JSON View (Auto Synced)

---

CRITICAL EXTRACTION INTELLIGENCE:

You MUST detect document type automatically:

CASE 1 — SINGLE PERSON DOCUMENT

Example:

Resume
Profile
ID Card

Output:

ONE RECORD ONLY

Example CSV:

Record,SNo,Name,Email,Phone,Title

1,1,MUSTAFA ALTHAF AZEEZ,althafazeez16@gmail.com,+91 7338502290,Electronics Engineer


CASE 2 — MULTIPLE PERSON DOCUMENT

Example:

HR Contact List

Output:

MULTIPLE RECORDS

Example CSV:

Record,SNo,Name,Email,Title,Company

1,1,Akanksha Puri,akanksha.puri@sourcefuse.com,Associate Director HR,SourceFuse

2,2,Akhil Jogiparthi,akhil@ibhubs.co,Vice President,iB Hubs

---

LEFT PANEL REQUIREMENT — CSV EDITOR

Render extracted data as:

Editable Table

Features:

Scrollable
Editable cells
Add row
Delete row

Columns auto detected

---

RIGHT PANEL REQUIREMENT — JSON VIEW

Render JSON version.

Example:

[
 {
  "Name": "Mustafa",
  "Email": "althafazeez16@gmail.com"
 }
]

---

AUTOSYNC REQUIREMENT (CRITICAL):

When user edits CSV table:

JSON MUST update instantly.

Example:

CSV change:

Phone → 9999999999

JSON auto update:

"Phone": "9999999999"

---

KEY VALUE EXTRACTION RULE:

Extract ALL fields.

Example:

Name:
Email:
Phone:
LinkedIn:
Skills:
Projects:

DO NOT SKIP ANYTHING

---

KEY VALUE OUTPUT MODE:

Internally maintain:

Key: Value

Example:

Name: MUSTAFA ALTHAF AZEEZ
Email: althafazeez16@gmail.com

---

AUTO SCHEMA DETECTION:

Columns auto detect from document.

DO NOT hardcode schema.

---

NULL HANDLING:

If missing:

Field = NULL

---

SYNC STORAGE:

CSV File:

output.csv

JSON File:

output.json

---

FINAL UI STRUCTURE:

LEFT:

CSV EDITOR

RIGHT:

JSON VIEWER

---

FINAL COMMAND:

Extract document.

Detect single or multiple records.

Generate CSV editor.

Generate JSON viewer.

Enable AutoSync CSV ↔ JSON.

Ensure 100% Key Value extraction.'''

# Use regex to find the prompt_str = """...""" block and replace it
pattern = re.compile(r'prompt_str\s*=\s*"""(.*?)"""', re.DOTALL)
content = pattern.sub(f'prompt_str = """\\n{new_prompt}\\n"""', content)

with open("extractor.py", "w") as f:
    f.write(content)

print("Prompt successfully patched!")
