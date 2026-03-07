import os
import io
import json
import xml.etree.ElementTree as ET
import pandas as pd
import pdfplumber
from PIL import Image
from typing import List, Optional
from models.pydantic_models import Document, Page, Block, BoundingBox

class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: str) -> Document:
        doc = Document(filename=os.path.basename(file_path))
        full_text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, pdf_page in enumerate(pdf.pages):
                page = Page(
                    page_number=i + 1,
                    width=float(pdf_page.width),
                    height=float(pdf_page.height)
                )
                # Extract embedded text directly from PDF (much more reliable than OCR for digital PDFs)
                page_text = pdf_page.extract_text() or ""
                page.metadata["embedded_text"] = page_text
                
                # Also extract tables if present
                tables = pdf_page.extract_tables()
                if tables:
                    page.metadata["tables"] = tables
                    
                full_text_parts.append(page_text)
                doc.pages.append(page)
                
        doc.metadata["full_text"] = "\n".join(full_text_parts)
        return doc

    @staticmethod
    def load_image(file_path: str) -> Document:
        img = Image.open(file_path)
        doc = Document(filename=os.path.basename(file_path))
        page = Page(
            page_number=1,
            width=float(img.width),
            height=float(img.height)
        )
        doc.pages.append(page)
        return doc
        
    @staticmethod
    def load_structured(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return pd.read_csv(file_path).to_json(orient='records')
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path).to_json(orient='records')
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f), indent=2)
        elif ext == '.xml':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read() # XML schema inference can be handled directly via LLM
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
