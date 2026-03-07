import json
import uuid
import pandas as pd
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from models.pydantic_models import Document, Page
from parser.document_loader import DocumentLoader
from parser.ocr import OCRProcessor
from parser.layout_analyzer import LayoutAnalyzer
from schema_generator import SchemaGenerator
from extractor import Extractor
from models.dynamic_models import DynamicModelGenerator
from utils.table_merger import TableMerger
from utils.post_processor import PostProcessor
from vectorstore.chroma_store import ChromaStore
from vectorstore.session_vector_registry import SessionVectorRegistry

class GraphState(TypedDict):
    session_id: str
    file_path: str
    document: Optional[Document]
    raw_text: str
    schema: Optional[Dict[str, Any]]
    extracted_data: Optional[Dict[str, Any]]
    doc_mode: Optional[str]   # TABLE_MODE | HIERARCHICAL_MODE | KEY_VALUE_MODE | ENTITY_LIST_MODE
    tabular_preview: Optional[Dict[str, Any]]   # set by TABULARIZER_NODE
    errors: List[str]
    iteration: int
    status: str
    orch_mode: Optional[str]
    orch_target: Optional[str]
    orch_execute: Optional[bool]
    orch_cloud: Optional[str]

def document_loader_node(state: GraphState) -> GraphState:
    state["status"] = "1/9 Document Loader"
    file_path = state["file_path"].lower()
    
    if file_path.endswith((".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")):
        if file_path.endswith(".pdf"):
            doc = DocumentLoader.load_pdf(state["file_path"])
        else:
            doc = DocumentLoader.load_image(state["file_path"])
        return {**state, "document": doc}
    else:
        # Structured file: CSV, Excel, JSON, XML, TXT
        text = DocumentLoader.load_structured(state["file_path"])
        return {**state, "raw_text": text}

def ocr_node(state: GraphState) -> GraphState:
    state["status"] = "2/9 OCR Agent"
    if not state.get("document"):
        return state # Skip if structured
    
    doc = state["document"]
    
    # Skip OCR entirely if we already have clean embedded text from pdfplumber
    # OCR (Tesseract) on digital PDFs produces garbled text; pdfplumber is far superior
    if doc.metadata.get("full_text", "").strip():
        return state
    
    file_path = state["file_path"]
    
    import fitz
    from PIL import Image
    
    is_pdf = file_path.lower().endswith(".pdf")
    if is_pdf:
        pdf_doc = fitz.open(file_path)
        
    for page in doc.pages:
        if is_pdf:
            page_obj = pdf_doc[page.page_number - 1]
            pix = page_obj.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            img = Image.open(file_path)
            
        tokens = OCRProcessor.extract_text_with_bboxes(img)
        # Store tokens in page for layout analyzer
        page.metadata["tokens"] = [t.model_dump() for t in tokens]
        
    if is_pdf:
        pdf_doc.close()
        
    return {**state, "document": doc}

def layout_analyzer_node(state: GraphState) -> GraphState:
    state["status"] = "3/9 Layout Analyzer"
    doc = state.get("document")
    if not doc:
        return state
    
    # Priority 1: Use embedded text extracted directly from PDF via pdfplumber (much more reliable)
    full_text = doc.metadata.get("full_text", "").strip()
    
    if full_text:
        # We have clean embedded text from pdfplumber - use it directly
        return {**state, "document": doc, "raw_text": full_text}
    
    # Priority 2: Fall back to OCR tokens if no embedded text was available (scanned PDFs / images)
    from models.pydantic_models import Token
    ocr_text = ""
    for page in doc.pages:
        tokens_data = page.metadata.get("tokens", [])
        tokens = [Token(**t) for t in tokens_data]
        LayoutAnalyzer.analyze_page_layout(page, tokens)
        ocr_text += "\n".join([block.text for block in page.blocks if block.text])
        
    return {**state, "document": doc, "raw_text": ocr_text}

def schema_generator_node(state: GraphState) -> GraphState:
    state["status"] = "4/9 Schema Generator Agent"
    try:
        from utils.strict_llm_client import StrictLLMClient
        from langchain_core.prompts import ChatPromptTemplate
        
        sample_text = state["raw_text"][:6000]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an Enterprise Document Schema Generator. Analyze the layout of this document. Identify all data points, text fields, table columns, and nested structures. Extract a JSON dictionary representing these keys. Provide empty strings for values. OUTPUT ONLY VALID JSON."),
            ("user", f"DOCUMENT CONTENT:\n{sample_text}")
        ])
        
        llm = StrictLLMClient(require_json=True)
        response = llm.invoke(prompt, {})
        
        if response.get("status") in ["success", "repaired"]:
            state["schema"] = response.get("data", {})
        else:
            state["schema"] = {}
            
    except Exception as e:
        state["errors"].append(f"Schema Gen Failed: {str(e)}")
        state["schema"] = {}
        
    return state

def extraction_node(state: GraphState) -> GraphState:
    state["status"] = "5/9 Extraction Agent"
    try:
        extractor = Extractor()

        current_errors = list(state.get("errors", []))

        orch_kwargs = {
            "mode": state.get("orch_mode", "EXTRACT_KV"),
            "target_table": state.get("orch_target", ""),
            "execute_permission": state.get("orch_execute", False),
            "cloud_provider": state.get("orch_cloud", "")
        }

        result = extractor.extract_data(state["raw_text"], None, current_errors, **orch_kwargs)

        # Aurora v2: capture both data and detected doc_mode
        state["extracted_data"] = result.get("data", {})
        state["doc_mode"] = result.get("doc_mode", "TABLE_MODE")
        print(f"[Aurora v2] Extraction complete. Mode: {state['doc_mode']}")

        state["errors"] = current_errors

    except Exception as e:
        state["errors"].append(f"Hard Panic in Extraction: {str(e)}")

    return state

def validation_agent_node(state: GraphState) -> GraphState:
    state["status"] = "6/9 Validation Agent"
    data = state.get("extracted_data")
    
    # Always increment iteration to prevent infinite recursion if should_retry is triggered
    state["iteration"] = state.get("iteration", 0) + 1
    
    # Check if extracted_data is None (not just empty dict)
    if data is None:
        state["errors"].append("Extraction failed to produce data.")
        return state
        
    try:
        if not isinstance(data, (dict, list)):
            raise ValueError(f"Extracted data is not a valid JSON object or array, got: {type(data)}")
        
        if len(data) == 0:
            # Empty dict is not ideal but not a crash - log a warning but pass
            state["status"] = "Validation Passed (Warning: Empty data)"
        else:
            state["status"] = "Validation Passed"
    except Exception as e:
        state["errors"].append(f"Validation Error: {str(e)}")
        state["status"] = "Validation Failed - Repairing"
    
    return state

def should_retry(state: GraphState):
    if state["errors"] and state["iteration"] < 2:
        return "extraction_node"
    return "semantic_completion_node"

def semantic_completion_node(state: GraphState) -> GraphState:
    state["status"] = "6.5/9 Semantic Completer (Implicit Fields)"
    
    # We run semantic completion if we have raw text, even if base extraction returned {}
    if state.get("raw_text"):
        current_data = state.get("extracted_data")
        if current_data is None:
            current_data = {}
            
        from utils.semantic_completer import SemanticCompleter
        print("[Aurora v2] Running Semantic Completer for universal implicit entities...")
        updated_data = SemanticCompleter.complete(state["raw_text"], current_data)
        state["extracted_data"] = updated_data
        
    return state

def paragraph_reconstruction_node(state: GraphState) -> GraphState:
    state["status"] = "6.8/9 Paragraph Reconstructor (OCR Context Sync)"
    
    if state.get("raw_text") and state.get("extracted_data"):
        from utils.paragraph_reconstructor import ParagraphReconstructor
        print("[Aurora v2] Running Paragraph Reconstructor for truncated fields...")
        updated_data = ParagraphReconstructor.reconstruct(state["raw_text"], state["extracted_data"])
        state["extracted_data"] = updated_data
        
    return state

def table_merger_node(state: GraphState) -> GraphState:
    state["status"] = "7/9 Table Merger Agent (Bypassed for True KV Mode)"
    # Anti-Gravity Project eliminates columnar pivots. 
    # Data is already natively extracted as True Row-Wise Records.
    return state

def post_processing_node(state: GraphState) -> GraphState:
    state["status"] = "8/9 Post-Processing Agent"
    if state.get("extracted_data"):
        processor = PostProcessor()
        clean_data = processor.normalize(state["extracted_data"])
        state["extracted_data"] = clean_data
    return state

def vector_store_node(state: GraphState) -> GraphState:
    state["status"] = "9/9 Vector Store Embedder Agent"
    if state.get("extracted_data") and state.get("session_id"):
        store = SessionVectorRegistry.get_store(state["session_id"])
        str_data = json.dumps(state["extracted_data"])
        filename = state["file_path"].split("/")[-1]
        store.add_documents([str_data], [{"filename": filename}], [str(uuid.uuid4())])
        import logging
        logging.getLogger(__name__).info(
            f"[VectorStore] Wrote {store.count()} vectors for session={state['session_id']}"
        )
    return state


def tabularizer_node(state: GraphState) -> GraphState:
    """Observer node — reads extracted_data, builds tabular_preview if confident.
    NEVER modifies extracted_data or any other canonical field."""
    state["status"] = "8.5/9 Tabularizer (Auto-Detect)"
    data = state.get("extracted_data")
    if not data:
        state["tabular_preview"] = None
        return state
    try:
        from utils.tabularizer import TabularDetector, Tabularizer
        is_tab, conf, reason = TabularDetector.detect(data)
        if is_tab:
            print(f"[Tabularizer] Tabular pattern detected (conf={conf:.2f}): {reason}")
            preview = Tabularizer.build_preview(data, confidence=conf)
            state["tabular_preview"] = preview
        else:
            print(f"[Tabularizer] Not tabular (conf={conf:.2f}): {reason}")
            state["tabular_preview"] = None
    except Exception as e:
        print(f"[Tabularizer] Error (non-fatal): {e}")
        state["tabular_preview"] = None
    return state

# Core Graph Builder
builder = StateGraph(GraphState)

builder.add_node("document_loader_node", document_loader_node)
builder.add_node("ocr_node", ocr_node)
builder.add_node("layout_analyzer_node", layout_analyzer_node)
builder.add_node("schema_generator_node", schema_generator_node)
builder.add_node("extraction_node", extraction_node)
builder.add_node("validation_agent_node", validation_agent_node)
builder.add_node("semantic_completion_node", semantic_completion_node)
builder.add_node("paragraph_reconstruction_node", paragraph_reconstruction_node)
builder.add_node("table_merger_node", table_merger_node)
builder.add_node("post_processing_node", post_processing_node)
builder.add_node("tabularizer_node", tabularizer_node)
builder.add_node("vector_store_node", vector_store_node)

builder.set_entry_point("document_loader_node")
builder.add_edge("document_loader_node", "ocr_node")
builder.add_edge("ocr_node", "layout_analyzer_node")
builder.add_edge("layout_analyzer_node", "schema_generator_node")
builder.add_edge("schema_generator_node", "extraction_node")
builder.add_edge("extraction_node", "validation_agent_node")

builder.add_conditional_edges("validation_agent_node", should_retry, {
    "extraction_node": "extraction_node",
    "semantic_completion_node": "semantic_completion_node"
})

builder.add_edge("semantic_completion_node", "paragraph_reconstruction_node")
builder.add_edge("paragraph_reconstruction_node", "table_merger_node")
builder.add_edge("table_merger_node", "post_processing_node")
builder.add_edge("post_processing_node", "tabularizer_node")
builder.add_edge("tabularizer_node", "vector_store_node")
builder.add_edge("vector_store_node", END)

workflow = builder.compile()
