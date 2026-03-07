import streamlit as st
import pandas as pd
import json
import os
import tempfile
import uuid
import io
import sqlite3

# --- PHASE 16: STRICT STARTUP VALIDATION ---
from config import settings
from dotenv import load_dotenv
load_dotenv()

print("\n=== SYSTEM BOOT: VALIDATING API KEYS ===")
required_keys = ["GROQ_API_KEY", "GEMINI_API_KEY"]
for key in required_keys:
    val = os.getenv(key) or getattr(settings, key, None)
    if not val:
        raise ValueError(f"CRITICAL ERROR: {key} not loaded. Please configure your .env file.")
    print(f"{key} loaded successfully.")
print("========================================\n")

# import asyncio
# import nest_asyncio

# Configure global event loop for LangChain/Streamlit threading compatibility
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# Apply nest_asyncio cleanly
# nest_asyncio.apply()

from extractor_graph import workflow
from database.sql_engine import SQLEngine
from vectorstore.chroma_store import ChromaStore, RAGAssistant
from vectorstore.session_vector_registry import SessionVectorRegistry
from cloud.universal_storage import UniversalStorageConnector
from utils.sync_engine import SyncEngine
from schema_generator import SchemaGenerator
from database.universal_connector import UniversalConnector
from utils.orchestration_engine import (
    reset_tabular_state,
    compute_available_export_formats,
    fire_master_data_updated_event,
    append_audit_entry,
)
from utils.cloud_push_validator import (
    validate_credentials as validate_cloud_credentials,
    map_push_error,
    generate_push_filename,
)
from ui.json_viewer import render_json_viewer

# --- SYSTEM INITIALIZATION ---
st.set_page_config(
    page_title="Aurora AI | Enterprise Document Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Canonical State Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "canonical_data" not in st.session_state:
    st.session_state.canonical_data = {}
if "df" not in st.session_state:
    st.session_state.df = None
if "csv_df" not in st.session_state:  # legacy alias
    st.session_state.csv_df = None
if "json_data" not in st.session_state:
    st.session_state.json_data = None
if "schema" not in st.session_state:
    st.session_state.schema = {}
if "schema_list" not in st.session_state:
    st.session_state.schema_list = []  # [{"field": ..., "type": ...}, ...]
if "doc_mode" not in st.session_state:
    st.session_state.doc_mode = "TABLE_MODE"  # TABLE_MODE | HIERARCHICAL_MODE | KEY_VALUE_MODE
if "status_logs" not in st.session_state:
    st.session_state.status_logs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = SessionVectorRegistry.get_store(st.session_state.session_id)
if "vector_persisted" not in st.session_state:
    st.session_state.vector_persisted = False  # opt-in persistence flag
if "uploaded_files_state" not in st.session_state:
    st.session_state.uploaded_files_state = {}
if "chatbot_edit_mode" not in st.session_state:
    st.session_state.chatbot_edit_mode = False
if "staged_changes" not in st.session_state:
    st.session_state.staged_changes = []
if "edit_history" not in st.session_state:
    st.session_state.edit_history = []
if "current_file_path" not in st.session_state:
    st.session_state.current_file_path = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Sync Engine Variables
if "auto_sync_enabled" not in st.session_state:
    st.session_state.auto_sync_enabled = True
if "sync_status" not in st.session_state:
    st.session_state.sync_status = ":material/sync_disabled: Not Synced"
if "is_syncing" not in st.session_state:
    st.session_state.is_syncing = False

# Aurora v2 — Document-specific table tracking
if "active_table" not in st.session_state:
    st.session_state.active_table = "document"   # set per-document at load time
if "master_json" not in st.session_state:
    st.session_state.master_json = None           # Layer 1: original extraction — NEVER overwritten by SQL
if "master_df" not in st.session_state:
    st.session_state.master_df = None             # Layer 2: editable table view of master_json
if "sql_result_df" not in st.session_state:
    st.session_state.sql_result_df = None         # Layer 3: temporary SQL query output only
if "disabled_cols" not in st.session_state:
    st.session_state.disabled_cols = []
# ── Tabularization module (additive, non-destructive) ─────────────────────────
if "tabular_preview" not in st.session_state:
    st.session_state.tabular_preview = None       # Set by TABULARIZER_NODE when pattern detected
if "tabular_mode_on" not in st.session_state:
    st.session_state.tabular_mode_on = False      # User-controlled toggle
if "available_export_formats" not in st.session_state:
    st.session_state.available_export_formats = ["JSON"]  # Single source of truth for export options

# --- UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Base Dark Theme Variables & Typography */
    :root {
        --primary-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        --text-gradient: linear-gradient(to right, #60a5fa, #c084fc);
        --glass-bg: rgba(15, 23, 42, 0.7);
        --glass-border: rgba(255, 255, 255, 0.08);
        --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        --bg-color: #0b0f19;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }
    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif !important; 
        color: var(--text-main) !important;
    }
    
    /* Main App Background (Dark Gradient) & Fade-In */
    .stApp { 
        background-color: var(--bg-color) !important;
        background-image: 
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.15), transparent 40%),
            radial-gradient(circle at bottom left, rgba(59, 130, 246, 0.15), transparent 40%) !important;
        animation: fadeInApp 1s ease-out forwards;
    }

    /* Animated Glow Header Text for "Aurora Intelligence" */
    @keyframes glowText {
        0% { background-position: 0% 50%; text-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
        50% { background-position: 100% 50%; text-shadow: 0 0 30px rgba(139, 92, 246, 0.6); }
        100% { background-position: 0% 50%; text-shadow: 0 0 20px rgba(59, 130, 246, 0.3); }
    }
    /* We target the specific markdown header using Streamlit's class structure */
    [data-testid="stSidebar"] h3 {
        background: var(--text-gradient);
        background-size: 200% auto;
        color: transparent !important;
        -webkit-background-clip: text;
        background-clip: text;
        animation: glowText 4s ease infinite;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        margin-bottom: 0px !important;
    }
    
    /* Top Header Override (Make it invisible to blend with background) */
    [data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    
    /* Left Sidebar - Fully Visible, Fixed Width, Dark Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border-right: 1px solid var(--glass-border) !important;
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.2) !important;
        
        /* FIX ALIGNMENT / CUTOFF: Force strict 300px width */
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        overflow-x: hidden !important;
        overflow-y: auto !important;
        
        animation: slideInLeft 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        z-index: 100 !important;
    }
    /* Ensure content inside doesn't overflow horizontally */
    [data-testid="stSidebarContent"] {
        width: 100% !important;
        padding-right: 10px !important; 
    }

    /* Main Content Container Formatting & Animation */
    [data-testid="stMainBlockContainer"] {
        padding-top: 3rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
        /* Slide up sequentially */
        opacity: 0;
        animation: slideUpContent 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.3s forwards;
    }

    /* Headings */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Buttons - Premium Dark Gradient, Smooth Hover */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: var(--text-main) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        position: relative;
        overflow: hidden;
    }
    /* Subtle button inner glow */
    .stButton > button::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(rgba(255,255,255,0.05), transparent);
        pointer-events: none; border-radius: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
        border-color: rgba(96, 165, 250, 0.4) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) scale(0.98) !important;
    }
    /* Primary buttons (Analyze, Push) */
    button[data-baseweb="button"][kind="primary"] {
        background: var(--accent-gradient) !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(96, 165, 250, 0.3) !important;
    }
    button[data-baseweb="button"][kind="primary"]:hover {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        box-shadow: 0 8px 30px rgba(96, 165, 250, 0.5) !important;
    }

    /* Input Fields & Selectors (Dark Glass) */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input,
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(10px);
        color: var(--text-main) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    /* Focus states */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > div:focus-within,
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }
    /* File Uploader styling */
    [data-testid="stFileUploaderDropzone"] {
        padding: 2.5rem !important;
        border: 2px dashed rgba(96, 165, 250, 0.3) !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {
        fill: #60a5fa !important;
    }

    /* Dataframe / Table Container (Dark Glass) */
    [data-testid="stDataFrame"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: var(--glass-shadow) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        width: 100% !important;
        opacity: 0;
        animation: fadeInCards 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.5s forwards;
    }
    [data-testid="stDataFrame"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    
    /* Table Headers */
    th[role="columnheader"] {
        background-color: rgba(30, 41, 59, 0.9) !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important;
        padding-top: 12px !important;
        padding-bottom: 12px !important;
    }
    /* Table Rows */
    td[role="gridcell"] {
        color: #e2e8f0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        transition: background-color 0.2s ease !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    /* Alternating Row Color & Hover */
    tr[role="row"]:nth-child(even) td[role="gridcell"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
    }
    tr[role="row"]:hover td[role="gridcell"] {
        background-color: rgba(96, 165, 250, 0.1) !important;
    }
    
    /* Tabs System UI */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 6px;
        gap: 8px;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 500 !important;
        color: var(--text-muted) !important;
        transition: all 0.3s ease !important;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-main) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #60a5fa !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    /* Hide the default underline bar in tabs */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* Terminal/Log Window (Intensified Glassmorphism) */
    .log-window {
        background: rgba(9, 14, 23, 0.8); 
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        color: #10b981; /* Matrix green text */
        font-family: 'Fira Code', 'Courier New', monospace;
        padding: 20px; 
        border-radius: 16px; 
        height: 200px; 
        overflow-y: auto; 
        font-size: 0.85rem;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5), 0 8px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
        border-top: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        opacity: 0;
        animation: fadeInCards 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.6s forwards;
    }
    .log-window:hover {
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5), 0 12px 40px rgba(0,0,0,0.4);
        border-color: rgba(96, 165, 250, 0.2);
    }
    
    /* JSON viewer wrapper styling */
    .stJson {
        background: rgba(15, 23, 42, 0.4) !important;
        border-radius: 12px !important;
        padding: 10px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }

    /* Status Badges */
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        backdrop-filter: blur(4px);
    }
    .badge-info { background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-success { background: rgba(52, 211, 153, 0.1); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-warning { background: rgba(251, 191, 36, 0.1); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }

    /* Core Layout Fixing Rules & Margins */
    /* Remove padding that causes horizontal scroll */
    .block-container {
        padding-bottom: 4rem !important;
    }
    /* Ensure markdown text inherits correct color */
    .stMarkdown p, .stMarkdown li {
        color: var(--text-muted);
        line-height: 1.6;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInApp {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    @keyframes slideInLeft {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideUpContent {
        0% { transform: translateY(40px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    @keyframes fadeInCards {
        0% { opacity: 0; transform: translateY(20px) scale(0.98); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    /* Custom Scrollbar for dark theme */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    ::-webkit-scrollbar-corner {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### :material/diamond: AURORA INTELLIGENCE")
    st.caption("v1.0.0 Production Edition")
    st.divider()
    
    st.subheader("📁 Data Ingestion")
    ingest_type = st.radio("Source Type", ["File Upload", "Database Pull", "Cloud Storage Pull"])
    
    upload_ready = False
    if ingest_type == "File Upload":
        uploaded_file = st.file_uploader("Upload File (PDF, Image, CSV, Excel, JSON)", type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx", "xls", "json", "xml"])
        if uploaded_file:
            if st.button("Load File to Workspace", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    st.session_state.current_file_path = tmp.name
                    upload_ready = True
                    st.success(f"Loaded {uploaded_file.name}")
                    
    elif ingest_type == "Database Pull":
        st.markdown("**Connect to Database**")
        db_plat = st.selectbox("Platform", ["PostgreSQL", "MySQL", "MongoDB", "Snowflake", "BigQuery"])
        db_creds = {}
        if db_plat in ["PostgreSQL", "MySQL"]:
            db_creds["host"] = st.text_input("Host")
            db_creds["port"] = st.text_input("Port", value="5432" if db_plat=="PostgreSQL" else "3306")
            db_creds["database"] = st.text_input("Database Name")
            db_creds["username"] = st.text_input("Username")
            db_creds["password"] = st.text_input("Password", type="password")
            db_creds["uri"] = st.text_input("OR Full Connection URI", type="password")
        elif db_plat == "MongoDB":
            db_creds["connection_uri"] = st.text_input("Connection URI", type="password")
            db_creds["database"] = st.text_input("Database Name")
            
        target_tb = st.text_input("Target Table / Collection")
        if st.button("Pull Data", use_container_width=True):
            with st.spinner("Connecting & Pulling..."):
                json_data = UniversalConnector.pull_database(db_plat, db_creds, target_tb)
                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
                    tmp.write(json_data)
                    st.session_state.current_file_path = tmp.name
                    upload_ready = True
                    st.success("Data pulled successfully!")

    elif ingest_type == "Cloud Storage Pull":
        st.markdown("**Connect to Cloud Storage**")
        cloud_plat = st.selectbox("Platform", ["AWS S3", "GCS", "Azure Blob"])
        cl_creds = {}
        if cloud_plat == "AWS S3":
            cl_creds["access_key"] = st.text_input("Access Key", type="password")
            cl_creds["secret_key"] = st.text_input("Secret Key", type="password")
            cl_creds["region"] = st.text_input("Region", value="us-east-1")
        elif cloud_plat == "GCS":
            cl_creds["service_account_json"] = st.text_area("Service Account JSON (paste full JSON)", height=200, placeholder='{"type": "service_account", ...}')
        elif cloud_plat == "Azure Blob":
            cl_creds["connection_string"] = st.text_input("Connection String", type="password")
            
        container_name = st.text_input("Bucket / Container Name")
        if st.button("List Files"):
            files = UniversalStorageConnector.list_files(cloud_plat, cl_creds, container_name)
            if files: st.session_state.cloud_files = files
            else: st.warning("No files found or access denied.")
            
        if "cloud_files" in st.session_state:
            target_file = st.selectbox("Select File", st.session_state.cloud_files)
            if st.button("Pull Selected File", use_container_width=True):
                with st.spinner("Downloading..."):
                    path = UniversalStorageConnector.pull_file(cloud_plat, cl_creds, container_name, target_file)
                    if path:
                        st.session_state.current_file_path = path
                        upload_ready = True
                        st.success("File downloaded.")
                    else:
                        st.error("Failed to download file.")

    st.divider()
    st.markdown("### :material/manufacturing: Action Orchestrator")
    st.session_state.orch_mode = st.selectbox(
        "Operation Mode", 
        ["EXTRACT_KV", "TO_SQL", "EXECUTE_SQL", "PUSH_CLOUD", "PULL_CLOUD", "PUSH_VECTOR", "PULL_VECTOR", "CHAT_RESPONSE", "WEBHOOK_NOTIFY"]
    )
    # Auto-populate from active_table so UI always reflects the current document
    _default_tbl = st.session_state.get("active_table", "document")
    st.session_state.orch_target = st.text_input(
        "Target SQL Table / Source Resource",
        value=_default_tbl,
        help="Auto-set from document filename. You can override manually."
    )
    if st.session_state.get("active_table") != st.session_state.orch_target:
        st.caption(f"Active Table: `{st.session_state.active_table}`")
    st.session_state.orch_cloud = st.selectbox("Target Cloud Provider", ["aws", "gcp", "azure"])
    st.session_state.orch_execute = st.checkbox("Grant Execute Permission (Destructive)", value=False)
    
    st.divider()
    nav = st.radio("Navigation View", ["Pipeline Dashboard", "Data Export (Push)"])
    st.divider()

    # ── Vector Session Status Badge ───────────────────────────────────────
    if st.session_state.get("vector_persisted"):
        st.success(":material/save: **Vectors: Saved Permanently**", icon=None)
    else:
        st.info(":material/bolt: **Vectors: Ephemeral** (session only)")

    # ── Clear Session ─────────────────────────────────────────────────────
    if st.button(":material/delete: Clear Session", help="Hard-delete all vectors and reset workspace", use_container_width=True):
        _sid = st.session_state.session_id
        audit = SessionVectorRegistry.clear_store(_sid)
        print(f"[ClearSession] Audit: {audit}")
        try:
            SyncEngine.auto_cleanup()
        except Exception:
            pass
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()



# --- MAIN DASHBOARD ---
if nav == "Pipeline Dashboard":
    st.markdown('<h1>Document Analysis Engine</h1>', unsafe_allow_html=True)
    
    col_input, col_status = st.columns([1, 1])
    with col_input:
        # If stuck in a processing state without active loop, clear it for user.
        if st.session_state.is_processing:
            if st.button(":material/power_off: Unlock Pipeline (Engine Stuck)"):
                st.session_state.is_processing = False
                st.rerun()

        analyze_clicked = st.button(
            "START INTELLIGENT PIPELINE", 
            type="primary", 
            use_container_width=True, 
            disabled=st.session_state.current_file_path is None
        )
        if analyze_clicked and st.session_state.current_file_path:
            st.session_state.is_processing = True
            status_indicator = st.empty()
            progress_bar = st.progress(0)

            # ══════════════════════════════════════════════════════════════════
            # CRITICAL FIX 1: FULL STATE RESET ON NEW DOCUMENT LOAD
            # Centralized via OrchestrationEngine — clears ALL table/SQL/CSV/
            # schema state so nothing persists from the previous document.
            # ══════════════════════════════════════════════════════════════════
            reset_tabular_state()
            # ── Full session reset for new document ──
            st.session_state.chat_history = []
            st.session_state.edit_history = []
            st.session_state.staged_changes = []
            st.session_state.chatbot_edit_mode = False
            st.session_state.document_id = str(uuid.uuid4())
            print(f"[SessionReset] New document_id: {st.session_state.document_id}")
            # ══════════════════════════════════════════════════════════════════
            
            try:
                initial_state = {
                    "session_id": st.session_state.session_id,
                    "file_path": st.session_state.current_file_path, "document": None, "raw_text": "", 
                    "schema": None, "extracted_data": None, "errors": [], "iteration": 0, "status": "Initializing",
                    "orch_mode": st.session_state.orch_mode,
                    "orch_target": st.session_state.orch_target,
                    "orch_execute": st.session_state.orch_execute,
                    "orch_cloud": st.session_state.orch_cloud
                }
                
                st.session_state.status_logs = []
                steps_count = 9 
                for i, output in enumerate(workflow.stream(initial_state)):
                    for node, state in output.items():
                        log_entry = f"[{node.upper()}] {state.get('status', 'Processed')}"
                        st.session_state.status_logs.append(log_entry)
                        
                        if state.get("schema") is not None: st.session_state.schema = state["schema"]
                        if state.get("doc_mode") is not None: st.session_state.doc_mode = state["doc_mode"]
                        # Capture tabular preview from TABULARIZER_NODE (additive, non-destructive)
                        if "tabular_preview" in state:
                            if state["tabular_preview"] is not None:
                                st.session_state.tabular_preview = state["tabular_preview"]
                                _tp = state["tabular_preview"]
                                _tp_conf = _tp.get("confidence", 0)
                                if _tp_conf >= 0.85:
                                    # Auto-activate tabular mode for high-confidence tables
                                    st.session_state.tabular_mode_on = True
                                    # CRITICAL FIX 4: SQL LIVE TABLE MODE
                                    st.session_state.sql_mode = "LIVE_TABLE"
                                    # Auto-populate master_df from tabular_preview rows
                                    _tp_rows = _tp.get("rows", [])
                                    if _tp_rows:
                                        import pandas as _pd
                                        _tab_df = _pd.DataFrame(_tp_rows)
                                        # Hide internal columns from UI
                                        _visible_cols = [c for c in _tab_df.columns if not str(c).startswith("__")]
                                        _tab_visible = _tab_df[_visible_cols].copy()
                                        st.session_state.master_df = _tab_visible
                                        st.session_state.df        = _tab_visible
                                        st.session_state.csv_df    = _tab_visible
                                        print(f"[TabularAuto] Auto-activated table mode ({_tp_conf:.2f}), {len(_tp_rows)} rows")
                                        # Compute export formats via DataFrame cell inspection (NOT tabular_mode toggle)
                                        compute_available_export_formats(df=_tab_visible)
                                        # Fire broadcast via orchestration engine
                                        fire_master_data_updated_event(source="TABULAR_AUTO_DETECT")
                                    else:
                                        # No rows — default to JSON-only
                                        compute_available_export_formats(df=None)

                                    # Auto table naming
                                    from utils.tabularizer import Tabularizer
                                    _raw_data = state.get("extracted_data") or {}
                                    _auto_tbl_name = Tabularizer.infer_table_name(_raw_data)
                                    st.session_state.active_table = _auto_tbl_name
                                    st.session_state.orch_target  = _auto_tbl_name
                                    print(f"[TabularAuto] Table name: {_auto_tbl_name} | SQL Mode: LIVE_TABLE")
                                else:
                                    st.session_state.tabular_mode_on = False
                                    st.session_state.sql_mode = "JSON"
                                    compute_available_export_formats(df=st.session_state.get("master_df"))

                        if state.get("extracted_data") is not None:
                            st.session_state.canonical_data = state["extracted_data"]

                            # State Hydration - preserve original data, no modification
                            raw = state["extracted_data"]
                            doc_mode = state.get("doc_mode", "TABLE_MODE")

                            # Auto-generate document-specific table name from file + mode
                            _file_path = st.session_state.get("current_file_path") or ""
                            _auto_tbl = SyncEngine.generate_table_name(_file_path, doc_mode)
                            st.session_state.active_table = _auto_tbl
                            st.session_state.orch_target = _auto_tbl
                            print(f"[Aurora v2] active_table set to: '{_auto_tbl}'")

                            if doc_mode in ("HIERARCHICAL_MODE", "KEY_VALUE_MODE", "ENTITY_LIST_MODE"):
                                # ── Layer 1: MASTER JSON — never overwritten by SQL
                                obj = raw if isinstance(raw, dict) else (raw[0] if isinstance(raw, list) and raw else {})
                                st.session_state.master_json = obj
                                st.session_state.json_data   = obj
                                # Schema from hierarchy
                                st.session_state.schema_list = SchemaGenerator.derive_auto(obj, doc_mode)
                                print(f"[Aurora v2] master_json stored ({len(obj)} top-level keys).")

                                # ── Layer 2: MASTER_DF via ProjectionKernel v2 (never touches master_json)
                                from utils.projection_kernel import ProjectionKernel, PROJECTION_LAYER
                                _proj_df, _proj_audit = ProjectionKernel.project(
                                    obj, doc_mode, projection_layer=PROJECTION_LAYER)
                                print(f"[ProjectionKernel] {_proj_audit['last_action']} | title: {_proj_audit['document_title']}")
                                if not _proj_df.empty:
                                    st.session_state.master_df = _proj_df
                                    st.session_state.df        = _proj_df
                                    st.session_state.csv_df    = _proj_df
                                    if _proj_audit.get("validation_warnings"):
                                        for w in _proj_audit["validation_warnings"]:
                                            print(f"[ProjectionKernel WARN] {w}")
                                else:
                                    # Kernel returned empty or failed — scalar fallback
                                    _reason = _proj_audit.get("fallback_reason")
                                    if _reason:
                                        st.warning(f":material/info: Table view unavailable; showing JSON. ({_reason})")
                                    scalar_row = {k: v for k, v in obj.items() if not isinstance(v, (list, dict))}
                                    _mdf = pd.DataFrame([scalar_row]) if scalar_row else pd.DataFrame()
                                    st.session_state.master_df = _mdf
                                    st.session_state.df        = _mdf
                                    st.session_state.csv_df    = _mdf

                                # Expose document title as separate metadata field for UI
                                if _proj_audit.get("document_title"):
                                    st.session_state.doc_title = _proj_audit["document_title"]

                                # ── Layer 3: SQL_RESULT_DF — clear on new document
                                st.session_state.sql_result_df = None
                            else:
                                # TABLE_MODE: flat list of records -> DataFrame via ProjectionKernel
                                data = raw if isinstance(raw, list) else [raw]
                                # ── Layer 1: MASTER JSON
                                st.session_state.master_json = data
                                st.session_state.json_data   = data

                                from utils.projection_kernel import ProjectionKernel, PROJECTION_LAYER
                                _proj_df, _proj_audit = ProjectionKernel.project(
                                    data, "TABLE_MODE", projection_layer=PROJECTION_LAYER)
                                print(f"[ProjectionKernel TABLE] {_proj_audit['last_action']}")

                                df = _proj_df if not _proj_df.empty else pd.DataFrame(data)

                                def _is_empty_col(series):
                                    _null_set = {"null", "none", "n/a", "na", "nan", ""}
                                    def _is_empty_val(v):
                                        if v is None: return True
                                        import numpy as np
                                        if isinstance(v, (list, dict, tuple)):
                                            return len(v) == 0
                                        if isinstance(v, np.ndarray):
                                            return v.size == 0
                                        if isinstance(v, str):
                                            return v.strip().lower() in _null_set
                                        try:
                                            return pd.isna(v)
                                        except Exception:
                                            return False
                                    return series.apply(_is_empty_val).all()

                                _cols_to_drop = [col for col in df.columns if _is_empty_col(df[col])]
                                if _cols_to_drop:
                                    print(f"[app.py] Dropping all-NULL columns: {_cols_to_drop}")
                                    df = df.drop(columns=_cols_to_drop)

                                if "SNo" in df.columns:
                                    print(f"[Extraction] Raw SNo from OCR: {list(df['SNo'])}")

                                st.session_state.schema_list = SchemaGenerator.derive_auto(data, "TABLE_MODE")
                                print(f"[Aurora v2] Table schema derived: {st.session_state.schema_list}")

                                # ── Layer 2: MASTER_DF — source of truth for CSV editor & sync
                                st.session_state.master_df = df
                                st.session_state.df        = df
                                st.session_state.csv_df    = df
                                # ── Layer 3: SQL_RESULT_DF — clear on new document
                                st.session_state.sql_result_df = None
                            
                        # Surface critical API Failures to the UI Log
                        if state.get("errors"):
                            for err in state["errors"]:
                                if err not in st.session_state.status_logs: # prevent duplicates
                                    st.session_state.status_logs.append(f"<span style='color: #ff4444;'>[FATAL ERROR] {str(err)}</span>")
                        
                    progress_bar.progress(min(1.0, (i + 1) / steps_count))
                
                st.success("Analysis Complete - Vectors Indexed")
            except Exception as e:
                import traceback
                st.error(f"Critical System Failure: {str(e)}")
                st.code(traceback.format_exc())
            finally:
                st.session_state.is_processing = False

    with col_status:
        st.subheader(":material/account_tree: Live LangGraph Execution")
        log_html = "".join([f"<div>{log}</div>" for log in st.session_state.status_logs])
        st.markdown(f'<div class="log-window">{log_html}</div>', unsafe_allow_html=True)

    st.divider()

    # Aurora v2: document-specific, schema-safe master sync
    def execute_master_sync(source: str, force: bool = False):
        if st.session_state.is_syncing: return
        if not st.session_state.auto_sync_enabled and not force: return

        # CRITICAL FIX: Hierarchical Mode now uses the SQL Engine.
        # It relies on master_json being converted into a relational view here.
        # So we explicitly allow HIERARCHICAL_MODE to sync to SQL.

        st.session_state.is_syncing = True
        print(f"\n--- SYNC STARTED [Source: {source}] ---")
        try:
            st.session_state.sync_status = ":material/sync: Syncing..."

            df = st.session_state.df
            if df is None or df.empty:
                print("[Autosync] No dataframe to sync — skipping.")
                st.session_state.sync_status = ":material/sync_disabled: No Data"
                return

            st.session_state.csv_df = df

            # Always use the document-specific active_table — never hardcode 'users'
            target_tb = st.session_state.get("active_table") or st.session_state.orch_target or "document"

            # Re-derive schema from current df
            current_records = df.to_dict(orient="records")
            current_schema = SchemaGenerator.derive_auto(current_records, "TABLE_MODE")
            if current_schema:
                st.session_state.schema_list = current_schema

            SyncEngine.sync_all(df, target_tb, schema=st.session_state.get('schema_list', []))

            sql_data = SyncEngine.fetch_all(target_tb)
            print(f"[Autosync] Rows in '{target_tb}' after sync: {len(sql_data)}")

            # JSON viewer stays in sync with df for TABLE_MODE
            st.session_state.json_data = df.to_dict(orient="records")

            st.session_state.sync_status = ":material/check_circle: Synced"
            print("SYNC COMPLETE")
        except Exception as e:
            import traceback
            print(f"SYNC ERROR: {e}")
            traceback.print_exc()
            st.session_state.sync_status = ":material/error: Sync Error"
        finally:
            st.session_state.is_syncing = False

    if st.session_state.df is not None:
        if "disabled_cols" not in st.session_state:
            st.session_state.disabled_cols = []
        st.markdown("<br><h3>Master Data & Schema</h3>", unsafe_allow_html=True)
        
        # Premium UI CSS for Status Card
        st.markdown('''
        <style>
        .premium-status {
            background: linear-gradient(90deg, rgba(16,185,129,0.1) 0%, rgba(16,185,129,0.02) 100%);
            border: 1px solid rgba(16,185,129,0.2);
            border-radius: 8px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 1.5rem;
        }
        .premium-status.manual {
            background: linear-gradient(90deg, rgba(245,158,11,0.1) 0%, rgba(245,158,11,0.02) 100%);
            border: 1px solid rgba(245,158,11,0.2);
        }
        .glow-dot {
            width: 10px;
            height: 10px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse-green 2s infinite;
        }
        .glow-dot.manual {
            background-color: #f59e0b;
            box-shadow: 0 0 10px #f59e0b;
            animation: pulse-orange 2s infinite;
        }
        @keyframes pulse-green {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        @keyframes pulse-orange {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }
        .status-text {
            color: #e2e8f0;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.5px;
        }
        </style>
        ''', unsafe_allow_html=True)
        
        # Controls Row (Spans full width above columns)
        ctrl1, ctrl_space, ctrl2 = st.columns([2, 8, 2])
        with ctrl1:
            st.session_state.auto_sync_enabled = st.toggle("Auto Sync", value=st.session_state.auto_sync_enabled, key="autosync")
        with ctrl2:
            sync_now = st.button(":material/sync: Sync Now", use_container_width=True)
            
        if sync_now:
            execute_master_sync("MANUAL_BUTTON", force=True)
            st.rerun()

        # Status Card Row
        if st.session_state.auto_sync_enabled:
            st.markdown('''
            <div class="premium-status">
                <div class="glow-dot"></div>
                <div class="status-text">SYSTEM STATUS: AUTO SYNC ACTIVE — Database Matrix Online</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="premium-status manual">
                <div class="glow-dot manual"></div>
                <div class="status-text">SYSTEM STATUS: MANUAL SYNC MODE — Pending Explicit Commit</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

        # -- Aurora v2 Mode Badge -----------------------------------------
        _mode = st.session_state.get("doc_mode", "TABLE_MODE")
        _mode_colors = {
            "TABLE_MODE":        ("#0f4", "#001f10"),
            "HIERARCHICAL_MODE": ("#4af", "#00111f"),
            "KEY_VALUE_MODE":    ("#fa4", "#1f0e00"),
        }
        _mc, _mbg = _mode_colors.get(_mode, ("#fff", "#111"))
        st.markdown(f"""
        <div style="display:inline-flex;align-items:center;gap:10px;
                    background:{_mbg};border:1px solid {_mc}44;border-radius:8px;
                    padding:6px 16px;margin-bottom:1.2rem;">
            <span style="width:9px;height:9px;border-radius:50%;background:{_mc};display:inline-block;"></span>
            <span style="color:{_mc};font-size:0.8rem;font-weight:700;letter-spacing:.12em;">
                AURORA v2 &nbsp;|&nbsp; {_mode.replace('_', ' ')}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if _mode in ("HIERARCHICAL_MODE", "KEY_VALUE_MODE"):
            # ── Hierarchical / Key-Value View ─────────────────────────────
            h1, h2 = st.columns([3, 2])
            with h1:
                _view_json = st.session_state.get("master_json") or st.session_state.json_data
                _tab_exists = st.session_state.get("tabular_mode_on", False)
                _tab_rows = None
                if _tab_exists:
                    _mdf = st.session_state.get("master_df")
                    if _mdf is not None and not _mdf.empty:
                        _tab_rows = _mdf.to_dict(orient="records")
                render_json_viewer(
                    _view_json,
                    tabular_state_exists=_tab_exists,
                    table_rows=_tab_rows,
                    container_height=720,
                    key_prefix="hier_json_viewer",
                )

                # ── Tabular Mode toggle (additive observer, non-destructive) ──────
                _tab_preview = st.session_state.get("tabular_preview")
                if _tab_preview and _tab_preview.get("confidence", 0) >= 0.85:
                    _tab_conf = _tab_preview["confidence"]
                    
                    st.toggle(
                        f":material/table_chart: Tabular Mode  (Conf: {_tab_conf:.2f})",
                        key="tabular_mode_on",
                        help="Auto-detected tabular structure. Toggle ON to open editable grid view."
                    )
                    
                    if st.session_state.get("tabular_mode_on"):
                        # Re-evaluate from actual DataFrame — toggle is display-only, not authoritative
                        compute_available_export_formats(df=st.session_state.get("master_df"))
                        _master = st.session_state.get("master_json") or st.session_state.json_data
                        from ui.tabular_viewer import render_tabular_viewer
                        render_tabular_viewer(_tab_preview, _master)
                    else:
                        # Re-evaluate from actual DataFrame even when toggled off
                        compute_available_export_formats(df=st.session_state.get("master_df"))


            with h2:
                st.markdown('''
                <style>
                .schema-header { font-size: 0.85rem; color: #94a3b8; font-weight: 600;
                                 margin-bottom: 0.5rem; padding-left: 5px; }
                </style>
                ''', unsafe_allow_html=True)

                # ── Schema Editor header + Add Field button ──────────────
                hse1, hse2 = st.columns([6, 3])
                with hse1:
                    st.markdown("#### Schema Editor")
                with hse2:
                    if st.button(":material/add: Add Field", use_container_width=True, key="hier_add_field_btn"):
                        new_field = f"New_Field_{len(st.session_state.schema_list) + 1}"
                        st.session_state.schema_list.append({"field": new_field, "type": "string"})
                        st.rerun()

                # ── Editable Schema rows ─────────────────────────────────
                with st.container(height=390):
                    hc1, hc2, hc3, hc4 = st.columns([4, 3, 2, 2])
                    hc1.markdown("<div class='schema-header'>Field Name</div>", unsafe_allow_html=True)
                    hc2.markdown("<div class='schema-header'>Data Type</div>", unsafe_allow_html=True)
                    hc3.markdown("<div class='schema-header'>Editable</div>", unsafe_allow_html=True)
                    hc4.markdown("<div class='schema-header'>Actions</div>", unsafe_allow_html=True)

                    def _hier_sync():
                        """Persist schema_list to disk and update sync status."""
                        _atbl = st.session_state.get("active_table", "document")
                        SyncEngine.sync_schema(st.session_state.schema_list, _atbl)
                        st.session_state.sync_status = ":material/check_circle: Synced"

                    # Helper to mutate the actual JSON object safely
                    def _mutate_json(action: str, old_key: str = "", new_key: str = ""):
                        _obj = st.session_state.get("master_json")
                        if not isinstance(_obj, dict): return
                        
                        if action == "rename" and old_key in _obj and new_key:
                            # Rebuild dict to preserve key order
                            _new_obj = {}
                            for k, v in _obj.items():
                                _new_obj[new_key if k == old_key else k] = v
                            st.session_state.master_json = _new_obj
                            st.session_state.json_data   = _new_obj
                        elif action == "delete" and old_key in _obj:
                            del _obj[old_key]
                            st.session_state.json_data = _obj
                        elif action == "add" and new_key:
                            _obj[new_key] = ""
                            st.session_state.json_data = _obj

                    _schema_types = ["string", "integer", "float", "boolean", "date", "array", "object", "array[object]"]
                    for _si, _srow in enumerate(st.session_state.schema_list):
                        _field    = _srow.get("field", "")
                        _ftype    = _srow.get("type", "string")
                        _editable = _srow.get("editable", True)
                        
                        # Generate a truly unique key for Streamlit widgets using both index and field name
                        _unique_key = f"{_si}_{_field}"

                        with st.container(border=True):
                            rc1, rc2, rc3, rc4 = st.columns([4, 3, 2, 2])

                            with rc3:
                                _new_edit = st.checkbox("Edit", value=_editable,
                                                        key=f"hier_edit_{_unique_key}", label_visibility="collapsed")
                                if _new_edit != _editable:
                                    st.session_state.schema_list[_si]["editable"] = _new_edit
                                    _hier_sync()
                                    st.rerun()

                            with rc1:
                                _new_name = st.text_input("Name", value=_field,
                                                          key=f"hier_name_{_unique_key}", label_visibility="collapsed",
                                                          disabled=not _editable)
                                if _new_name != _field and _new_name.strip():
                                    _new_clean = _new_name.strip()
                                    _mutate_json("rename", old_key=_field, new_key=_new_clean)
                                    st.session_state.schema_list[_si]["field"] = _new_clean
                                    _hier_sync()
                                    st.rerun()

                            with rc2:
                                _tidx = _schema_types.index(_ftype) if _ftype in _schema_types else 0
                                _new_type = st.selectbox("Type", _schema_types, index=_tidx,
                                                         key=f"hier_type_{_unique_key}", label_visibility="collapsed",
                                                         disabled=not _editable)
                                if _new_type != _ftype:
                                    st.session_state.schema_list[_si]["type"] = _new_type
                                    _hier_sync()
                                    st.rerun()

                            with rc4:
                                with st.popover(":material/delete:", use_container_width=True):
                                    st.markdown(f"Confirm delete **{_field}**?")
                                    if st.button("Delete", key=f"hier_del_{_unique_key}",
                                                 type="primary", use_container_width=True):
                                        _mutate_json("delete", old_key=_field)
                                        # Delete by exact index to avoid deleting OTHER fields with same name
                                        st.session_state.schema_list.pop(_si)
                                        _hier_sync()
                                        st.rerun()


                # ── Top-Level Fields (derived from live master_json — never stored separately) ──
                st.markdown("#### Top-Level Fields ↔")
                with st.container(height=200):
                    _top = st.session_state.get("master_json") or st.session_state.get("json_data")
                    if _top is None:
                        st.caption(":material/info: No document loaded yet.")
                    elif isinstance(_top, dict):
                        if not _top:
                            st.caption(":material/info: Empty document.")
                        for k, v in _top.items():
                            if isinstance(v, (list, dict)):
                                st.markdown(f"**{k}** `{type(v).__name__}` — {len(v)} items")
                            else:
                                _new_val = st.text_input(
                                    k, value=str(v),
                                    key=f"hfield_{k}",
                                    label_visibility="visible"
                                )
                                if _new_val != str(v):
                                    st.session_state.master_json[k] = _new_val
                                    st.session_state.json_data = st.session_state.master_json
                                    # Trigger global refresh so Table, Schema, Chatbot all update
                                    from utils.live_sync import refresh_all_views
                                    refresh_all_views(source="TOP_LEVEL_FIELD_EDIT")
                                    st.rerun()
                    elif isinstance(_top, list):
                        # TABLE_MODE: master_json is a list of records — derive summary
                        st.caption(f":material/table_chart: Flat table — {len(_top)} records")
                        if _top and isinstance(_top[0], dict):
                            _cols = list(_top[0].keys())
                            st.markdown(f"**Columns:** {', '.join(_cols[:10])}{'...' if len(_cols) > 10 else ''}")
                    else:
                        st.caption(f":material/info: Data type: `{type(_top).__name__}`")

        else:
            # ── TABLE_MODE: Standard CSV DataGrid ─────────────────────────
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### CSV Editor  *(master\_df — source of truth)*")

                _evt = st.session_state.get("MASTER_DATA_UPDATED_EVENT", 0)
                if st.session_state.get("_csv_last_global_event", -1) != _evt:
                    st.session_state.csv_has_pending_master_commit = False
                    st.session_state._csv_last_global_event = _evt

                if "csv_has_pending_master_commit" not in st.session_state:
                    st.session_state.csv_has_pending_master_commit = False

                c1a, c1b = st.columns([3, 1])
                with c1a:
                    if st.session_state.csv_has_pending_master_commit:
                        st.warning(":material/warning: Pending CSV edits.")
                with c1b:
                    if st.session_state.csv_has_pending_master_commit:
                        if st.button(":material/check_circle: Apply", type="primary", use_container_width=True):
                            from utils.live_sync import update_master_model
                            # Commit staged_csv_data → MASTER_DATA_MODEL
                            update_master_model(df=st.session_state.staged_csv_data.copy(), source="CSV_EDITOR_APPLY")
                            st.session_state.csv_has_pending_master_commit = False
                            if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                            st.session_state.grid_key_counter += 1
                            if st.session_state.auto_sync_enabled:
                                execute_master_sync("CSV_DATAGRID_APPLY")
                            st.rerun()

                # Always bind to master_df, EXCEPT during temporary edit session
                if st.session_state.csv_has_pending_master_commit and "staged_csv_data" in st.session_state:
                    _editor_source = st.session_state.staged_csv_data
                else:
                    _editor_source = st.session_state.get("master_df") if st.session_state.get("master_df") is not None else st.session_state.df

                edited_df = st.data_editor(
                    _editor_source,
                    use_container_width=True,
                    num_rows="dynamic",
                    hide_index=True,
                    disabled=st.session_state.disabled_cols,
                    key=f"csv_editor_{st.session_state.get('grid_key_counter', 0)}",
                    height=720
                )

                # Direct write mode to staged copy
                _changed = (
                    _editor_source.shape != edited_df.shape or
                    _editor_source.to_csv(index=False) != edited_df.to_csv(index=False)
                )
                if _changed and not st.session_state.is_syncing:
                    # STORE change in staged_csv_data AND MARK csv_has_pending_master_commit = TRUE
                    st.session_state.staged_csv_data = edited_df.copy()
                    st.session_state.csv_has_pending_master_commit = True
                    st.rerun()


            with c2:
                    st.markdown('''
                    <style>
                    .schema-header {
                        font-size: 0.85rem;
                        color: #94a3b8;
                        font-weight: 600;
                        margin-bottom: 0.5rem;
                        padding-left: 5px;
                    }
                    .schema-container form {
                        margin: 0;
                    }
                    </style>
                    ''', unsafe_allow_html=True)

                    sc1, sc2 = st.columns([6, 3])
                    with sc1:
                        st.markdown("#### Schema Editor")
                    with sc2:
                        if st.button(":material/add: Add Field", use_container_width=True):
                            new_col = f"New_Field_{len(st.session_state.master_df.columns)+1}"
                            while new_col in st.session_state.master_df.columns:
                                new_col += "_"
                            # ── Live Sync Update ──
                            _mdf = st.session_state.master_df.copy()
                            _mdf[new_col] = ""
                            from utils.live_sync import update_master_model
                            update_master_model(df=_mdf, source="SCHEMA_ADD_FIELD")
                            if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                            st.session_state.grid_key_counter += 1
                            st.rerun()

                    with st.container(height=390):
                        hc1, hc2, hc3, hc4 = st.columns([4, 3, 2, 2])
                        hc1.markdown("<div class='schema-header'>Field Name</div>", unsafe_allow_html=True)
                        hc2.markdown("<div class='schema-header'>Data Type</div>", unsafe_allow_html=True)
                        hc3.markdown("<div class='schema-header'>Editable</div>", unsafe_allow_html=True)
                        hc4.markdown("<div class='schema-header'>Actions</div>", unsafe_allow_html=True)

                        current_cols = list(st.session_state.master_df.columns)
                        for i, col_name in enumerate(current_cols):
                            with st.container(border=True):
                                rc1, rc2, rc3, rc4 = st.columns([4, 3, 2, 2])
                                is_disabled = col_name in st.session_state.disabled_cols

                                with rc3:
                                    edit_val = not is_disabled
                                    new_edit_val = st.checkbox("Edit", value=edit_val, key=f"edit_{st.session_state.get('grid_key_counter', 0)}_{i}", label_visibility="collapsed")
                                    if new_edit_val != edit_val:
                                        if new_edit_val and col_name in st.session_state.disabled_cols:
                                            st.session_state.disabled_cols.remove(col_name)
                                            if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                                            st.session_state.grid_key_counter += 1
                                        elif not new_edit_val and col_name not in st.session_state.disabled_cols:
                                            st.session_state.disabled_cols.append(col_name)
                                            if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                                            st.session_state.grid_key_counter += 1
                                        st.rerun()

                                with rc1:
                                    new_name = st.text_input("Name", value=col_name, key=f"name_{st.session_state.get('grid_key_counter', 0)}_{i}", label_visibility="collapsed", disabled=is_disabled)
                                    if new_name != col_name and new_name.strip() != "":
                                        if new_name not in st.session_state.master_df.columns:
                                            # ── Live Sync Update ──
                                            _mdf = st.session_state.master_df.copy()
                                            _mdf.rename(columns={col_name: new_name}, inplace=True)
                                            if col_name in st.session_state.disabled_cols:
                                                st.session_state.disabled_cols.remove(col_name)
                                                st.session_state.disabled_cols.append(new_name)
                                            from utils.live_sync import update_master_model
                                            update_master_model(df=_mdf, source="SCHEMA_RENAME")
                                            # Re-derive schema after rename
                                            st.session_state.schema_list = SchemaGenerator.derive_auto(st.session_state.json_data, "TABLE_MODE")
                                            if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                                            st.session_state.grid_key_counter += 1
                                            st.rerun()

                                with rc2:
                                    curr_type = "string"
                                    dtype = st.session_state.df[col_name].dtype
                                    import pandas as pd
                                    if pd.api.types.is_integer_dtype(dtype): curr_type = "integer"
                                    elif pd.api.types.is_float_dtype(dtype): curr_type = "float"
                                    elif pd.api.types.is_bool_dtype(dtype): curr_type = "boolean"
                                    elif pd.api.types.is_datetime64_any_dtype(dtype): curr_type = "date"

                                    type_opts = ["string", "integer", "float", "boolean", "date"]
                                    new_type = st.selectbox("Type", type_opts, index=type_opts.index(curr_type), key=f"type_{st.session_state.get('grid_key_counter', 0)}_{i}", label_visibility="collapsed", disabled=is_disabled)
                                    if new_type != curr_type:
                                        try:
                                            _mdf = st.session_state.master_df.copy()
                                            if new_type == "integer": _mdf[col_name] = pd.to_numeric(_mdf[col_name], errors='coerce').fillna(0).astype('Int64')
                                            elif new_type == "float": _mdf[col_name] = pd.to_numeric(_mdf[col_name], errors='coerce')
                                            elif new_type == "boolean": _mdf[col_name] = _mdf[col_name].astype(bool)
                                            elif new_type == "date": _mdf[col_name] = pd.to_datetime(_mdf[col_name], errors='coerce')
                                            else: _mdf[col_name] = _mdf[col_name].astype(str)

                                            from utils.live_sync import update_master_model
                                            update_master_model(df=_mdf, source="SCHEMA_TYPE_CHANGE")
                                            if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                                            st.session_state.grid_key_counter += 1
                                            st.rerun()
                                        except Exception:
                                            pass

                                with rc4:
                                    with st.popover(":material/delete:", use_container_width=True):
                                        st.markdown(f"Confirm delete **{col_name}**?")
                                        if st.button("Delete", key=f"del_{st.session_state.get('grid_key_counter', 0)}_{i}", type="primary", use_container_width=True):
                                            _mdf = st.session_state.master_df.copy()
                                            _mdf.drop(columns=[col_name], inplace=True)
                                            if col_name in st.session_state.disabled_cols:
                                                st.session_state.disabled_cols.remove(col_name)
                                            from utils.live_sync import update_master_model
                                            update_master_model(df=_mdf, source="SCHEMA_DEL_FIELD")
                                            # Re-derive schema after column delete
                                            st.session_state.schema_list = SchemaGenerator.derive_auto(st.session_state.json_data, "TABLE_MODE")
                                            if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                                            st.session_state.grid_key_counter += 1
                                            st.rerun()

                    st.markdown("<div style='margin-bottom: 22px;'></div>", unsafe_allow_html=True)
                    _table_json_data = st.session_state.json_data
                    _tab_rows_for_viewer = None
                    _mdf_viewer = st.session_state.get("master_df")
                    if _mdf_viewer is not None and not _mdf_viewer.empty:
                        _tab_rows_for_viewer = _mdf_viewer.to_dict(orient="records")
                    render_json_viewer(
                        _table_json_data,
                        tabular_state_exists=True,
                        table_rows=_tab_rows_for_viewer,
                        container_height=410,
                        key_prefix="table_json_viewer",
                    )

        st.divider()
        st.markdown("#### Export Data")

        import json as _json
        import io

        # ── LIVE RECOMPUTE: always use master_df at render time ─────────
        _live_df = st.session_state.get("master_df")
        _available_formats = compute_available_export_formats(df=_live_df)

        # ── Resolve master_json: dedicated key > json_data > canonical_data
        _master_json = (
            st.session_state.get("master_json")
            or st.session_state.get("json_data")
            or st.session_state.get("canonical_data")
        )

        if _available_formats == ["JSON"]:
            # ── HIERARCHICAL: JSON only ───────────────────────────────────
            _mode_label = st.session_state.get("doc_mode", "HIERARCHICAL MODE").replace("_", " ")
            st.caption(
                f"**{_mode_label}** detected — CSV and Excel export are disabled to prevent data loss. "
                "Nested fields (BillTo, ShipTo, LineItems, Tax, etc.) are preserved in JSON only."
            )
            if _master_json:
                _json_str = _json.dumps(_master_json, indent=2, ensure_ascii=False)
                st.download_button(
                    ":material/code: Export JSON  (Full Document — all nested fields)",
                    data=_json_str,
                    file_name="aurora_export.json",
                    mime="application/json",
                    key="json_download_hier_btn",
                    use_container_width=True
                )
            else:
                st.download_button(":material/code: Export JSON", data=b"", file_name="empty.json", disabled=True, use_container_width=True)


        else:
            # ── TABLE_MODE / TABULAR: CSV + Excel + JSON ──────────────────
            exc_col, csv_col, json_col = st.columns(3)
            _export_df = st.session_state.csv_df.copy() if st.session_state.csv_df is not None else pd.DataFrame()

            with exc_col:
                _output = io.BytesIO()
                if not _export_df.empty:
                    with pd.ExcelWriter(_output, engine='openpyxl') as _writer:
                        _export_df.to_excel(_writer, index=False)
                    st.download_button(
                        ":material/table_view: Export Excel",
                        data=_output.getvalue(),
                        file_name="aurora_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.download_button(":material/table_view: Export Excel", data=b"", file_name="empty.xlsx", disabled=True, use_container_width=True)

            with csv_col:
                if not _export_df.empty:
                    st.download_button(
                        ":material/description: Export CSV",
                        data=_export_df.to_csv(index=False).encode("utf-8"),
                        file_name="aurora_export.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.download_button(":material/description: Export CSV", data=b"", file_name="empty.csv", disabled=True, use_container_width=True)

            with json_col:
                if _master_json:
                    _json_str = _json.dumps(_master_json, indent=2, ensure_ascii=False)
                    st.download_button(
                        ":material/code: Export JSON",
                        data=_json_str,
                        file_name="aurora_export.json",
                        mime="application/json",
                        key="json_download_flat_btn",
                        use_container_width=True
                    )
                else:
                    st.download_button(":material/code: Export JSON", data=b"", file_name="empty.json", disabled=True, use_container_width=True)

    else:
        st.info("System idle. Process a document to view data.")

    # SQL Engine is now available for ALL modes (TABULAR + HIERARCHICAL)
    _tabs_list = [":material/psychology: Intelligence Hub", ":material/database: SQL Engine"]
    _rendered_tabs = st.tabs(_tabs_list)
    tab_discovery = _rendered_tabs[0]
    tab_sql = _rendered_tabs[1] if len(_rendered_tabs) > 1 else None

    with tab_discovery:
        col_chat_head1, col_chat_head2 = st.columns([1, 1])
        with col_chat_head1:
            st.subheader("Vector Grounded AI Chat")
        with col_chat_head2:
            # ── Chatbot Edit Lock UI with animated indicator ──
            _edit_mode = st.toggle(
                "Chatbot Edit Mode",
                value=st.session_state.chatbot_edit_mode,
                help="Grant chat agent permission to propose edits (staged). You must Apply to commit.",
                key="_chatbot_edit_toggle"
            )
            st.session_state.chatbot_edit_mode = _edit_mode
            if _edit_mode:
                st.markdown("""
                    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                    <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                        <span class="material-icons" style="color:#22c55e;font-size:20px;animation:chatbot-pulse 2s infinite;">lock_open</span>
                        <span style="color:#22c55e;font-size:13px;font-weight:600;">Chatbot Edit: ON</span>
                    </div>
                    <style>
                    @keyframes chatbot-pulse {
                        0% { transform: scale(1); opacity: 1; }
                        50% { transform: scale(1.12); opacity: 0.85; }
                        100% { transform: scale(1); opacity: 1; }
                    }
                    </style>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                    <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                        <span class="material-icons" style="color:#94a3b8;font-size:20px;">lock</span>
                        <span style="color:#94a3b8;font-size:13px;font-weight:600;">Chatbot Edit: OFF</span>
                    </div>
                """, unsafe_allow_html=True)
            if st.session_state.edit_history:
                with st.expander(":material/history: Edit History", expanded=False):
                    for h_idx, h in enumerate(reversed(st.session_state.edit_history)):
                        col_hist1, col_hist2 = st.columns([3, 1])
                        with col_hist1:
                            st.caption(f"**{h.get('timestamp')}**: {len(h.get('operations', []))} ops — {h.get('status')}")
                        with col_hist2:
                            if h.get('status') == "COMMITTED":
                                if st.button(":material/undo: Undo", key=f"undo_{h_idx}"):
                                    from utils.edit_orchestrator import undo_staged_changes
                                    _inv_ops = h.get('inverse_ops', [])
                                    if not _inv_ops:
                                        st.info(":material/info: Nothing to undo for this change")
                                    else:
                                        success, msg = undo_staged_changes(
                                            _inv_ops,
                                            st.session_state.get("master_json"),
                                            st.session_state.get("master_df")
                                        )
                                        if success:
                                            real_idx = len(st.session_state.edit_history) - 1 - h_idx
                                            st.session_state.edit_history[real_idx]["status"] = "UNDONE"
                                            from utils.orchestration_engine import fire_master_data_updated_event, append_audit_entry
                                            fire_master_data_updated_event(source="UNDO")
                                            append_audit_entry(
                                                actor="CHATBOT_UNDO",
                                                action="UNDO",
                                                operations=_inv_ops,
                                                status="UNDONE_COMMIT",
                                            )
                                            st.session_state.chat_history.append({"role": "assistant", "content": f":material/undo: **Undid a previous commit ({len(_inv_ops)} operations reverted).**"})
                                            st.success(":material/check_circle: Undo successful!")
                                            st.rerun()
                                        else:
                                            st.error(f":material/warning: Undo failed: {msg}")
        
        chat_box = st.container(height=400)

        def get_db_data(table: str = "agencies"):
            """FIX 5: Fetch live SQL data to pass to the chatbot LLM."""
            rows = SyncEngine.fetch_all(table)
            print(f"[Chatbot] get_db_data received {len(rows)} rows")
            return rows

        def chatbot_answer(question: str) -> str:
            """Fast single-call chatbot using LIVE session data as primary context.
            
            Supports TWO modes:
              - QUESTION mode: answers questions about the document
              - EDIT mode: proposes staged edits (only when edit mode ON + edit intent detected)
            """
            from utils.groq_client import get_groq_client
            from langchain_core.messages import SystemMessage, HumanMessage

            # ── INTENT DETECTION ──
            _edit_verbs = {
                "delete", "remove", "update", "change", "add", "insert",
                "rename", "set", "modify", "replace", "drop", "clear",
                "edit", "fix", "correct", "merge", "split", "swap",
            }
            _q_lower = question.lower().strip()
            _words = set(_q_lower.split())
            _is_edit_intent = bool(_words & _edit_verbs)
            print(f"[Chatbot] Intent: {'EDIT' if _is_edit_intent else 'QUESTION'} | Edit mode: {st.session_state.get('chatbot_edit_mode')}")

            # ── LIVE DATA BINDING (PRIORITY ORDER) ──
            # PRIORITY 1: master_json (SINGLE SOURCE OF TRUTH)
            _master_json = st.session_state.get("master_json") or st.session_state.get("json_data")

            # PRIORITY 2: table_dataframe (master_df) for tabular context
            _live_df = st.session_state.get("master_df")
            if _live_df is not None and not _live_df.empty:
                session_data = _live_df.to_dict(orient="records")
                print(f"[Chatbot] Using LIVE master_df ({len(session_data)} rows, cols: {list(_live_df.columns)})")
            elif _master_json:
                session_data = _master_json if isinstance(_master_json, list) else [_master_json]
                print(f"[Chatbot] Using master_json ({len(session_data) if isinstance(session_data, list) else 'dict'})")
            else:
                # SQL fallback
                session_data = get_db_data(st.session_state.get("orch_target") or "document")
                print(f"[Chatbot] Using SQL fallback ({len(session_data)} rows)")

            # Normalise to list
            if isinstance(session_data, dict):
                session_data = [session_data]
            elif not isinstance(session_data, list):
                session_data = []
            data_context = json.dumps(session_data[:200], indent=2) if session_data else "No extracted data available yet. Please process a document first."
            print(f"[Chatbot] Rows in context: {len(session_data)}")

            # ── FULL DOCUMENT CONTEXT (master_json as-is for hierarchical awareness) ──
            _master_json_str = ""
            if _master_json:
                try:
                    _master_json_str = json.dumps(_master_json, indent=2)[:8000]  # cap at 8K chars
                except Exception:
                    _master_json_str = str(_master_json)[:4000]

            # ── SCHEMA CONTEXT ──
            _live_columns = []
            if _live_df is not None and not _live_df.empty:
                _live_columns = list(_live_df.columns)
                _schema_lines = "\n".join(f"  - {c} ({str(_live_df[c].dtype)})" for c in _live_columns)
                schema_context = f"CURRENT SCHEMA (live from table — {len(_live_df)} rows):\n{_schema_lines}"
            else:
                _schema_list = st.session_state.get("schema_list") or []
                if _schema_list:
                    _live_columns = [s.get('field', '?') for s in _schema_list]
                    _schema_lines = "\n".join(
                        f"  - {s.get('field','?')} ({s.get('type','string')})"
                        for s in _schema_list
                    )
                    schema_context = f"CURRENT SCHEMA (from schema_list):\n{_schema_lines}"
                else:
                    schema_context = ""

            # ── EDIT HISTORY CONTEXT ──
            _edit_history = st.session_state.get("edit_history", [])
            _edit_summary = ""
            if _edit_history:
                _recent = _edit_history[-5:]  # last 5 commits
                _lines = []
                for h in _recent:
                    _lines.append(f"  - [{h.get('status')}] {h.get('timestamp', '?')}: {len(h.get('operations', []))} operations")
                _edit_summary = f"\nRECENT EDIT HISTORY ({len(_edit_history)} total commits):\n" + "\n".join(_lines)

            # ── TOP-LEVEL FIELDS CONTEXT ──
            _top_level = ""
            if isinstance(_master_json, dict):
                _scalars = {k: v for k, v in _master_json.items() if not isinstance(v, (list, dict))}
                if _scalars:
                    _top_level = "\nTOP-LEVEL DOCUMENT FIELDS:\n" + json.dumps(_scalars, indent=2)

            # ── BUILD SYSTEM PROMPT ──
            _doc_mode = st.session_state.get("doc_mode", "TABLE_MODE")
            system_prompt = f"""You are Aurora, an enterprise AI document assistant.
You have access to the FULL live extracted document data below.

IMPORTANT RULES:
- Answer ONLY from the data provided below.
- The schema shows the current field names; if a field was renamed or deleted, it is reflected here.
- If a field no longer exists in the schema or data, say so clearly.
- Do NOT reference field names that don't appear in the Current Schema.
- If the data doesn't contain the answer, say so briefly and honestly.
- You can answer questions about the document content, structure, values, totals, counts, etc.
- If the user asks what data is loaded, describe the document structure and key fields.

Document Mode: {_doc_mode}

{schema_context}
{_top_level}
{_edit_summary}

FULL DOCUMENT (master_json):
{_master_json_str}

TABLE DATA (flattened rows):
{data_context}
"""

            # ── ROUTING: EDIT vs QUESTION ──
            if st.session_state.get("chatbot_edit_mode") and _is_edit_intent:
                # EDIT MODE + EDIT INTENT → route to atomic edit engine
                from utils.chatbot_editor import generate_atomic_edits
                
                _schema_dict = {"columns": _live_columns} if _live_columns else {"columns": list(session_data[0].keys()) if session_data else []}
                
                print(f"[ChatbotEditor] Routing to edit engine: {question}")
                proposal = generate_atomic_edits(
                    user_request=question,
                    schema_context=_schema_dict,
                    sample_data=session_data
                )
                
                if proposal.clarification_needed:
                    return f":material/help: **Clarification needed:** {proposal.clarification_needed}"
                elif not proposal.operations:
                    return "I couldn't determine which exact columns or rows to edit. Could you be more specific?"
                else:
                    for op in proposal.operations:
                        op_dict = op.model_dump()
                        op_dict["_source_prompt"] = question
                        st.session_state.staged_changes.append(op_dict)
                    return ":material/check_circle: **Proposed Change Ready for Approval** — Review below and click Apply or Revert."
            else:
                # QUESTION MODE (or edit mode OFF) → answer with full context
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=question)
                ]

                try:
                    llm = get_groq_client("llama-3.1-8b-instant")
                    print("[Chatbot] Using model: llama-3.1-8b-instant")
                    response = llm.invoke(messages)
                    return response.content
                except Exception as e:
                    print(f"[Chatbot] Primary model failed: {e}, trying fallback...")
                    try:
                        llm = get_groq_client("gemma2-9b-it")
                        print("[Chatbot] Using model: gemma2-9b-it (fallback)")
                        response = llm.invoke(messages)
                        return response.content
                    except Exception as e2:
                        print(f"[Chatbot] Fallback also failed: {e2}")
                        return f"I encountered an error: {str(e2)}"

        # ── ALWAYS ENABLE CHAT INPUT ──
        user_input = st.chat_input("Query extracted intelligence...", key="discovery_chat_input")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                answer = chatbot_answer(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # ── ALL CHAT CONTENT INSIDE chat_box (scroll container) ──
        with chat_box:
            for m in st.session_state.chat_history:
                st.chat_message(m["role"]).write(m["content"])

            # ── STAGED CHANGES INTERCEPT PANEL (inside chat scroll) ──
            if st.session_state.get("staged_changes"):
                st.info(":material/info: **Proposed changes (staged) — review before applying**")
                with st.container(border=True):
                    for idx, op in enumerate(st.session_state.staged_changes):
                        _cond = op.get('condition', {})
                        _tgt_display = f"{_cond.get('column','?')} {_cond.get('operator','?')} {_cond.get('value','?')}" if _cond else str(op.get('target', {}))
                        st.markdown(f"**Op {idx+1}:** `{op['type']}` :material/arrow_forward: {_tgt_display}")
                        st.caption(f"*Reason:* {op.get('reason', 'N/A')}")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        @st.dialog(":material/warning: Destructive Change Confirmation")
                        def confirm_destructive_dialog():
                            st.warning(":material/warning: You are about to execute destructive operations. Please type **DELETE** to proceed.")
                            for _op in st.session_state.staged_changes:
                                if "DELETE" in _op['type']:
                                    _c = _op.get('condition') or {}
                                    _t = _op.get('target') or {}
                                    if _c:
                                        st.markdown(f"- `{_op['type']}`: {_c.get('column','?')} {_c.get('operator','?')} {_c.get('value','?')}")
                                    else:
                                        # Non-condition delete (e.g. DELETE_COLUMN)
                                        _desc = ', '.join(f"{k}: {v}" for k, v in _t.items()) if _t else "all"
                                        st.markdown(f"- `{_op['type']}`: {_desc}")
                            _ack = st.text_input("Type DELETE to confirm")
                            if st.button("Proceed", type="primary", disabled=_ack != "DELETE"):
                                from utils.edit_orchestrator import apply_staged_changes
                                success, msg, inverse_ops = apply_staged_changes(
                                    st.session_state.staged_changes,
                                    st.session_state.get("master_json"),
                                    st.session_state.get("master_df")
                                )
                                if success:
                                    from utils.orchestration_engine import fire_master_data_updated_event, append_audit_entry
                                    st.session_state.edit_history.append({
                                        "commit_id": str(uuid.uuid4())[:8],
                                        "timestamp": __import__('datetime').datetime.now().isoformat()[:19],
                                        "status": "COMMITTED",
                                        "operations": st.session_state.staged_changes.copy(),
                                        "inverse_ops": inverse_ops,
                                        "document_id": st.session_state.get("document_id", "unknown"),
                                        "session_id": st.session_state.get("session_id", "unknown"),
                                    })
                                    print(f"[Commit] destructive commit: {len(inverse_ops)} inverse ops")
                                    append_audit_entry(actor="CHATBOT", action="COMMIT", operations=st.session_state.staged_changes.copy(), status="COMMITTED")
                                    compute_available_export_formats(df=st.session_state.get("master_df"))
                                    # Sync aliases + SQLite + fire event (do NOT re-project via ProjectionKernel
                                    # as it normalizes column names and would revert renames)
                                    from utils.live_sync import broadcast_sync
                                    broadcast_sync(source="CHATBOT_EDIT_DESTRUCTIVE")
                                    fire_master_data_updated_event(source="CHATBOT_EDIT")
                                    st.session_state.canonical_data = st.session_state.get("master_json")
                                    # Bump grid_key_counter so Schema Editor text_inputs get fresh keys
                                    # (prevents stale widget state from triggering phantom reverse renames)
                                    if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                                    st.session_state.grid_key_counter += 1
                                    st.session_state.chat_history.append({"role": "assistant", "content": f":material/check_circle: **Applied {len(st.session_state.staged_changes)} destructive changes successfully.**"})
                                    st.session_state.staged_changes = []
                                    st.success(":material/check_circle: Change Applied Successfully")
                                    st.rerun()
                                else:
                                    st.error(f":material/warning: {msg}")
                        
                        if st.button(":material/check_circle: APPLY CHANGE", type="primary", use_container_width=True):
                            is_destructive = any("DELETE" in op.get("type", "") for op in st.session_state.staged_changes)
                            if is_destructive:
                                confirm_destructive_dialog()
                            else:
                                from utils.edit_orchestrator import apply_staged_changes
                                success, msg, inverse_ops = apply_staged_changes(
                                    st.session_state.staged_changes,
                                    st.session_state.get("master_json"),
                                    st.session_state.get("master_df")
                                )
                                if success:
                                    from utils.orchestration_engine import fire_master_data_updated_event, append_audit_entry
                                    st.session_state.edit_history.append({
                                        "commit_id": str(uuid.uuid4())[:8],
                                        "timestamp": __import__('datetime').datetime.now().isoformat()[:19],
                                        "status": "COMMITTED",
                                        "operations": st.session_state.staged_changes.copy(),
                                        "inverse_ops": inverse_ops,
                                        "document_id": st.session_state.get("document_id", "unknown"),
                                        "session_id": st.session_state.get("session_id", "unknown"),
                                    })
                                    print(f"[Commit] non-destructive commit: {len(inverse_ops)} inverse ops")
                                    append_audit_entry(actor="CHATBOT", action="COMMIT", operations=st.session_state.staged_changes.copy(), status="COMMITTED")
                                    compute_available_export_formats(df=st.session_state.get("master_df"))
                                    # Sync aliases + SQLite + fire event (do NOT re-project via ProjectionKernel
                                    # as it normalizes column names and would revert renames)
                                    from utils.live_sync import broadcast_sync
                                    broadcast_sync(source="CHATBOT_EDIT_APPLY")
                                    fire_master_data_updated_event(source="CHATBOT_EDIT")
                                    st.session_state.canonical_data = st.session_state.get("master_json")
                                    # Bump grid_key_counter so Schema Editor text_inputs get fresh keys
                                    # (prevents stale widget state from triggering phantom reverse renames)
                                    if "grid_key_counter" not in st.session_state: st.session_state.grid_key_counter = 0
                                    st.session_state.grid_key_counter += 1
                                    st.session_state.chat_history.append({"role": "assistant", "content": f":material/check_circle: **Applied {len(st.session_state.staged_changes)} changes successfully.**"})
                                    st.session_state.staged_changes = []
                                    st.success(":material/check_circle: Change Applied Successfully")
                                    st.rerun()
                                else:
                                    st.error(f":material/warning: {msg}")

                    with col_btn2:
                        if st.button(":material/undo: REVERT CHANGE", use_container_width=True):
                            from utils.orchestration_engine import append_audit_entry
                            st.session_state.edit_history.append({
                                "commit_id": str(uuid.uuid4())[:8],
                                "timestamp": __import__('datetime').datetime.now().isoformat()[:19],
                                "status": "REVERTED",
                                "operations": st.session_state.staged_changes.copy(),
                                "inverse_ops": [],
                                "document_id": st.session_state.get("document_id", "unknown"),
                                "session_id": st.session_state.get("session_id", "unknown"),
                            })
                            append_audit_entry(actor="CHATBOT", action="REVERT", operations=st.session_state.staged_changes.copy(), status="REVERTED")
                            st.session_state.chat_history.append({"role": "assistant", "content": ":material/undo: **Reverted proposed changes.**"})
                            st.session_state.staged_changes = []
                            st.success(":material/check_circle: Change Reverted Successfully")
                            st.rerun()

    if tab_sql:
     with tab_sql:
        sql_eng = SQLEngine()

        # ── Always use the session-state active_table — never fall back to DB tables
        # from a previous session. This is the ONLY source of truth for which document
        # is currently loaded.
        active_tbl = st.session_state.get("active_table") or st.session_state.get("orch_target") or "document"
        _sql_mode = st.session_state.get("sql_mode", "JSON")
        _mode_badge = ":material/table_chart: LIVE TABLE" if _sql_mode == "LIVE_TABLE" else ":material/code: JSON"

        st.subheader(f"NL-to-SQL Intelligence Engine  |  Active Table: `{active_tbl}`")
        st.caption(f"SQL Mode: **{_mode_badge}**")

        # ── If active_table doesn't exist in DB yet, or master_df is newer than DB,
        # force-sync current master_df to DB NOW so SQL always sees current data.
        _master_df_for_sql = (
            st.session_state.get("master_df")
            if st.session_state.get("master_df") is not None
            else st.session_state.get("csv_df")
        )
        _doc_mode_sql = st.session_state.get("doc_mode", "TABLE_MODE")

        if _master_df_for_sql is not None and not _master_df_for_sql.empty:
            # Force-sync current master_df to DB NOW so SQL always sees current data.
            # CRITICAL FIX: We sync for ALL MODES so SQL engine has its runtime relational view.
            SyncEngine.sync_all(_master_df_for_sql, active_tbl)

        # ── Show live table info
        if SyncEngine.table_exists(active_tbl):
            live_cols = sql_eng.get_columns(active_tbl)
            live_rows = SyncEngine.fetch_all(active_tbl)
            st.success(f"Table `{active_tbl}` — {len(live_rows)} rows  |  Columns: {', '.join(live_cols)}")
        else:
                # No hardcoded block for hierarchical mode anymore.
                # If table is not available, process a document or trigger sync.
                st.warning(f"Table `{active_tbl}` not available. Ensure AutoSync is enabled or click Sync.")

        # ── Clear stale SQL cache when active_table changes (new document loaded)
        if st.session_state.get("_sql_tab_active_tbl") != active_tbl:
            st.session_state._sql_tab_active_tbl = active_tbl
            st.session_state.last_nl_input  = ""
            st.session_state.generated_sql  = ""
            st.session_state.sql_result_df  = None
            print(f"[SQLTab] Document changed → cleared SQL cache. New table: {active_tbl}")

        nl_input = st.text_input(
            "Describe your query in plain English:",
            placeholder='e.g. "show all data" / "delete ShippingHandling" / "change Total to 500"',
            key="nl_sql_input"
        )

        if nl_input:
            if "last_nl_input" not in st.session_state or st.session_state.last_nl_input != nl_input:
                st.session_state.last_nl_input = nl_input
                with st.spinner("Generating SQL..."):
                    st.session_state.generated_sql = sql_eng.generate_sql(
                        nl_input,
                        st.session_state.schema,
                        table_hint=active_tbl
                    )

            generated_sql = st.session_state.get("generated_sql", "")

            if generated_sql.strip().upper().startswith("ERROR"):
                st.error(f"Cannot generate SQL: {generated_sql}")
            elif generated_sql:
                st.code(generated_sql, language="sql")
                if not generated_sql.strip().upper().startswith("SELECT"):
                    st.warning("This is a write operation — it will modify the database.")

                if st.button("Execute SQL Query", key="exec_sql_btn", type="primary"):
                    with st.spinner("Executing..."):
                        res, msg = sql_eng.execute_raw(generated_sql)

                    if "error" in msg.lower():
                        st.error(msg)
                    else:
                        st.caption(msg)

                    _is_select = generated_sql.strip().upper().startswith(("SELECT", "WITH"))

                    if _is_select:
                        # ── SELECT: show result in sql_result_df — master UNTOUCHED
                        if not res.empty:
                            st.session_state.sql_result_df = res
                            st.markdown("**SQL Result View** *(temporary — master data unchanged)*")
                            st.dataframe(res, hide_index=True, use_container_width=True)
                        else:
                            st.info("Query returned no rows.")
                    else:
                        # ── WRITE: committed to DB by execute_raw — now reload from DB
                        # Fresh connection, fresh cursor — no stale state
                        import sqlite3 as _sqlite3
                        _db_path = SyncEngine.get_db_path()
                        try:
                            _conn = _sqlite3.connect(_db_path, check_same_thread=False)
                            _cur  = _conn.cursor()
                            _conn.commit()   # ensure WAL is flushed
                            updated_df = pd.read_sql_query(
                                f'SELECT * FROM "{active_tbl}"', _conn
                            )
                            _conn.close()
                        except Exception as _e:
                            updated_df = pd.DataFrame()
                            st.error(f"Reload failed: {_e}")

                        if not updated_df.empty:
                            st.session_state.sql_result_df = updated_df

                        if st.session_state.auto_sync_enabled:
                            # ── AUTO SYNC ON: immediately promote to master + rerun
                            if not updated_df.empty:
                                from utils.live_sync import update_master_model
                                update_master_model(df=updated_df.copy(), source="SQL_AUTO_SYNC")
                                st.session_state.canonical_data = st.session_state.master_json
                                from schema_generator import SchemaGenerator
                                st.session_state.schema_list = SchemaGenerator.derive_auto(
                                    st.session_state.master_json, "TABLE_MODE"
                                )
                                if "grid_key_counter" not in st.session_state:
                                    st.session_state.grid_key_counter = 0
                                st.session_state.grid_key_counter += 1
                                st.session_state.sql_result_df = None
                                st.session_state.sync_status = ":material/check_circle: Synced"
                                st.success(f":material/check_circle: Auto-synced — {len(updated_df)} rows applied to CSV Editor, JSON Viewer and Schema.")
                            import time; time.sleep(0.4)
                            st.rerun()
                        else:
                            # ── AUTO SYNC OFF: show for review, explicit apply
                            if not updated_df.empty:
                                st.markdown("**SQL Result View** *(write applied to DB — review before applying to master)*")
                                st.dataframe(updated_df, hide_index=True, use_container_width=True)
                            st.divider()
                            st.warning(":material/warning: SQL applied to database. Master data is unchanged until you click Apply.")
                            if st.button(":material/check_circle: Apply SQL changes to Master Data", key="apply_sql_master_btn", type="primary"):
                                if st.session_state.sql_result_df is not None and not st.session_state.sql_result_df.empty:
                                    from utils.live_sync import update_master_model
                                    update_master_model(df=st.session_state.sql_result_df.copy(), source="SQL_MANUAL_APPLY")
                                    st.session_state.canonical_data = st.session_state.master_json
                                    from schema_generator import SchemaGenerator
                                    st.session_state.schema_list = SchemaGenerator.derive_auto(
                                        st.session_state.master_json, "TABLE_MODE"
                                    )
                                    if "grid_key_counter" not in st.session_state:
                                        st.session_state.grid_key_counter = 0
                                    st.session_state.grid_key_counter += 1
                                    st.session_state.sql_result_df = None
                                    st.session_state.sync_status = ":material/check_circle: Synced"
                                    st.success(f":material/check_circle: Master data updated — {len(st.session_state.master_df)} rows applied to all editors.")
                                    import time; time.sleep(0.4)
                                    st.rerun()



elif nav == "Data Export (Push)":
    st.header("Enterprise Cloud & Database Push Synchronization")
    if not st.session_state.canonical_data:
        st.warning("No canonical data in workspace. Please ingest and process data first.")
    else:
        st.write("Ready to push current canonical data to external systems.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Push to Database")
            db_plat = st.selectbox("Push Platform", ["PostgreSQL", "MySQL", "MongoDB"])
            db_creds = {}
            if db_plat in ["PostgreSQL", "MySQL"]:
                db_creds["uri"] = st.text_input("Full Connection URI", type="password", key="db_push_uri")
                db_creds["host"] = st.text_input("Host", key="db_push_h")
                db_creds["username"] = st.text_input("Username", key="db_push_u")
                db_creds["password"] = st.text_input("Password", type="password", key="db_push_p")
                db_creds["database"] = st.text_input("Database Name", key="db_push_d")
                db_creds["port"] = st.text_input("Port", value="5432" if db_plat=="PostgreSQL" else "3306", key="db_push_pt")
            elif db_plat == "MongoDB":
                db_creds["connection_uri"] = st.text_input("Connection URI", type="password", key="mongo_push_uri")
                db_creds["database"] = st.text_input("Database Name", key="mongo_push_db")
            
            target_tb = st.text_input("Destination Table / Collection")
            
            # ── Export format from shared OrchestrationEngine state ──
            _db_formats = st.session_state.get("available_export_formats", ["JSON"])
            
            export_format_db = st.selectbox("Export Format", _db_formats, index=0, key="db_export_format", help="Available formats come from Pipeline configuration.")
            if st.button(":material/database: Push to Database", use_container_width=True):
                # ── Pre-flight validation ──
                _missing = False
                if not target_tb: _missing = True
                if db_plat in ["PostgreSQL", "MySQL"] and not db_creds.get("uri"):
                    if not all([db_creds.get("host"), db_creds.get("username"), db_creds.get("password"), db_creds.get("database"), db_creds.get("port")]):
                        _missing = True
                elif db_plat == "MongoDB" and not db_creds.get("connection_uri"):
                    if not db_creds.get("database"):
                        _missing = True
                
                if _missing:
                    st.error(":material/warning: Please provide all required credentials before pushing.")
                else:
                    with st.spinner("Testing & Pushing..."):
                        try:
                            push_data = st.session_state.canonical_data
                            if export_format_db in ["CSV", "EXCEL"]:
                                import pandas as pd
                                push_data = pd.DataFrame(st.session_state.canonical_data)
                            
                            success, msg = UniversalConnector.push_database(db_plat, db_creds, target_tb, push_data)
                            
                            if success: 
                                st.success(":material/check_circle: Successfully pushed to database.")
                            else:
                                _m = str(msg).lower()
                                if "password authentication failed" in _m:
                                    st.error(":material/error: Incorrect password. Please verify your database password.")
                                elif "could not connect to server" in _m:
                                    st.error(":material/error: Database host unreachable. Check host and port.")
                                elif "database does not exist" in _m:
                                    st.error(":material/error: Database not found. Check database name.")
                                elif "authentication failed" in _m:
                                    st.error(":material/error: Invalid username or password.")
                                elif "timeout expired" in _m:
                                    st.error(":material/error: Connection timeout. Check network access.")
                                else:
                                    st.error(":material/error: Connection failed. Please verify credentials and try again.")
                        except Exception as e:
                            _e = str(e).lower()
                            if "password authentication failed" in _e:
                                st.error(":material/error: Incorrect password. Please verify your database password.")
                            elif "could not connect to server" in _e:
                                st.error(":material/error: Database host unreachable. Check host and port.")
                            elif "database does not exist" in _e:
                                st.error(":material/error: Database not found. Check database name.")
                            elif "authentication failed" in _e:
                                st.error(":material/error: Invalid username or password.")
                            elif "timeout expired" in _e:
                                st.error(":material/error: Connection timeout. Check network access.")
                            else:
                                st.error(":material/error: Connection failed. Please verify credentials and try again.")
                    
        with col2:
            st.subheader("Push to Cloud Storage")
            cloud_plat = st.selectbox("Cloud Platform", ["AWS S3", "GCS", "Azure Blob"])
            cl_creds = {}
            if cloud_plat == "AWS S3":
                cl_creds["access_key"] = st.text_input("Access Key", type="password", key="s3_ak")
                cl_creds["secret_key"] = st.text_input("Secret Key", type="password", key="s3_sk")
                cl_creds["region"] = st.text_input("Region", value="us-east-1", key="s3_rg")
            elif cloud_plat == "GCS":
                cl_creds["service_account_json"] = st.text_area("Service Account JSON (paste full JSON)", height=200, key="gcs_sa", placeholder='{"type": "service_account", ...}')
            elif cloud_plat == "Azure Blob":
                cl_creds["connection_string"] = st.text_input("Connection String", type="password", key="az_cs")
                
            container_name = st.text_input("Bucket / Container Name", key="c_push_bn")
            target_file_name = st.text_input("Destination Filename (leave blank for auto-generated)", key="c_push_fn")
            
            # ── Export format dropdown from OrchestrationEngine ──
            _cloud_formats = st.session_state.get("available_export_formats", ["JSON"])
            export_format_cloud = st.selectbox("Export Format", _cloud_formats, index=0, key="cloud_export_format", help="Available formats come from Pipeline configuration.")
            
            if st.button(":material/cloud_upload: Push to Cloud", use_container_width=True):
                # ── Pre-flight validation via CloudPushValidator ──
                _valid, _missing_fields = validate_cloud_credentials(cloud_plat, cl_creds, container_name)
                
                if not _valid:
                    st.error(f":material/error: Missing required fields: **{', '.join(_missing_fields)}**")
                else:
                    # ── Auto-generate filename if blank ──
                    if target_file_name and target_file_name.strip():
                        dest_name = target_file_name.strip()
                        if export_format_cloud == "JSON" and not dest_name.lower().endswith(".json"): dest_name += ".json"
                        elif export_format_cloud == "CSV" and not dest_name.lower().endswith(".csv"): dest_name += ".csv"
                        elif export_format_cloud == "EXCEL" and not dest_name.lower().endswith(".xlsx"): dest_name += ".xlsx"
                    else:
                        _doc_id = st.session_state.get("active_table") or st.session_state.get("orch_target") or "aurora_export"
                        dest_name = generate_push_filename(_doc_id, export_format_cloud)
                    
                    suffix = ".json"
                    if export_format_cloud == "CSV": suffix = ".csv"
                    elif export_format_cloud == "EXCEL": suffix = ".xlsx"
                    
                    import tempfile
                    import os
                    tmp_path = tempfile.mktemp(suffix=suffix)
                    
                    try:
                        if export_format_cloud == "JSON":
                            import json
                            with open(tmp_path, "w", encoding="utf-8") as f:
                                json.dump(st.session_state.canonical_data, f, indent=2)
                        elif export_format_cloud == "CSV":
                            import pandas as pd
                            pd.DataFrame(st.session_state.canonical_data).to_csv(tmp_path, index=False)
                        elif export_format_cloud == "EXCEL":
                            import pandas as pd
                            pd.DataFrame(st.session_state.canonical_data).to_excel(tmp_path, index=False)

                        with st.spinner(f":material/cloud_upload: Uploading {export_format_cloud} → {cloud_plat}..."):
                            success, msg = UniversalStorageConnector.push_file(cloud_plat, cl_creds, container_name, tmp_path, dest_name)
                            if success:
                                st.success(f":material/check_circle: Successfully pushed **{dest_name}** to {cloud_plat} / {container_name}")
                                # ── Audit trail entry ──
                                append_audit_entry(
                                    actor="CLOUD_PUSH",
                                    action=f"PUSH_{cloud_plat.upper().replace(' ', '_')}",
                                    operations=[{"type": "CLOUD_PUSH", "target": f"{cloud_plat}/{container_name}/{dest_name}", "format": export_format_cloud}],
                                    status="COMMITTED",
                                )
                            else:
                                # ── Friendly error via CloudPushValidator ──
                                _friendly = map_push_error(cloud_plat, str(msg))
                                st.error(f":material/error: {_friendly}")
                    except Exception as e:
                        _friendly = map_push_error(cloud_plat, str(e))
                        st.error(f":material/error: {_friendly}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
