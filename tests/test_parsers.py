import pytest
import os
import json
from services.parser_factory import ParserFactory
from core.models import Document

def test_parser_factory_logic():
    # Mocking files just to test factory selection
    with open("test.json", "w") as f:
        json.dump({"test": "data"}, f)
    
    parser = ParserFactory.get_parser("test.json")
    assert "StructuredParser" in str(type(parser))
    
    os.remove("test.json")

def test_json_parsing():
    content = {"invoice_no": "INV-100", "total": 500}
    with open("sample.json", "w") as f:
        json.dump(content, f)
        
    parser = ParserFactory.get_parser("sample.json")
    doc = parser.parse("sample.json")
    
    assert isinstance(doc, Document)
    assert doc.doc_type == "structured"
    assert len(doc.pages) == 1
    assert "invoice_no" in doc.pages[0].blocks[0].metadata["key"]
    
    os.remove("sample.json")
