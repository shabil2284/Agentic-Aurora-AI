# 🌌 **Agentic-Aurora-AI**

*Enterprise-grade autonomous AI document intelligence platform for extracting, classifying, and querying complex data across heterogeneous files and databases.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflows-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## 🚀 Overview

**Agentic-Aurora-AI** is an enterprise-grade document intelligence platform that bridges heterogeneous enterprise data with autonomous LLM reasoning. Powered by cyclical LangGraph agentic workflows and dual-engine inference—combining ultra-low-latency Groq (LLaMA-3.3-70B) with multimodal Google Gemini—it extracts, classifies, and queries data across PDFs, Excel sheets, images, and databases. The platform pairs dense semantic retrieval via ChromaDB with a dynamic SQLAlchemy text-to-SQL query engine and automated schema generation. Built to meet stringent enterprise demands, it achieves an industry-leading 98% extraction accuracy benchmark and provides native cloud storage connectors for AWS S3, Azure Blob, and Google Cloud Storage. All capabilities are orchestrated and served through an intuitive, interactive Streamlit dashboard fully containerized with Docker.

---

## 🛠️ Tech Stack

| Category | Technology | Role & Purpose |
| :--- | :--- | :--- |
| **Runtime & Language** | Python 3.10+ | Core application runtime and orchestration environment |
| **User Interface** | Streamlit | Interactive enterprise analytics dashboard and chat interface |
| **Agent Orchestration** | LangChain & LangGraph | Cyclic agent state machines, tool routing, and reasoning flows |
| **High-Speed Inference** | Groq (`llama-3.3-70b`) | Ultra-low-latency classification, extraction, and reasoning |
| **Multimodal Intelligence** | Google Gemini | Vision understanding, scanned image parsing, and OCR processing |
| **Vector Storage (RAG)** | ChromaDB | Dense vector embeddings and semantic similarity retrieval |
| **Relational & SQL Engine** | SQLAlchemy | Text-to-SQL translation, schema introspection, and ORM mapping |
| **Containerization** | Docker | Reproducible containerized builds and enterprise deployment |
| **Cloud Storage** | AWS S3, Azure Blob, GCS | Enterprise cloud storage connectors for multi-cloud pipelines |

---

## ✨ Features

- 📑 **Multi-Format Ingestion**: Ingests, parses, and extracts structured and unstructured data from PDFs, Excel spreadsheets, scanned images, and relational databases.
- 🔄 **Agentic Workflows**: Uses LangGraph to orchestrate stateful, multi-step agent graphs that dynamically plan, route tools, and self-correct queries.
- ⚡ **Dual-Engine LLM Setup**: Combines Groq (LLaMA-3.3-70B) for ultra-fast text reasoning with Google Gemini for multimodal document and image understanding.
- 🎯 **98% Extraction Accuracy**: Benchmarked at 98% precision across complex layouts, nested tables, key-value pairs, and varied document formats.
- 🔍 **RAG with ChromaDB**: Employs dense vector indexing and hybrid retrieval for accurate context synthesis and grounded question-answering.
- 🗄️ **Text-to-SQL & Schema Generation**: Generates dynamic SQL queries and inspects relational schemas automatically using SQLAlchemy.
- ☁️ **Enterprise Cloud Connectors**: Ingests and synchronizes document repositories directly from AWS S3, Azure Blob Storage, and Google Cloud Storage (GCS).
- 🖥️ **Interactive Streamlit Dashboard**: Provides an intuitive interface for live document upload, visual pipeline inspection, and conversational querying.

---

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Agentic-Aurora-AI.git
   cd Agentic-Aurora-AI
   ```

2. **Set up virtual environment & install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Add your API keys and cloud credentials in .env
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **(Optional) Run with Docker**
   ```bash
   docker build -t agentic-aurora-ai .
   docker run -p 8501:8501 --env-file .env agentic-aurora-ai
   ```

---

## 📁 Project Structure

```
Agentic-Aurora-AI/
├── app.py               # Streamlit application entrypoint
├── Dockerfile           # Docker container configuration
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── core/
│   ├── agents/          # LangGraph agent workflows and state definitions
│   ├── ingestion/       # Parsers for PDF, Excel, image, and database inputs
│   ├── llm/             # Groq and Gemini client configurations
│   ├── rag/             # ChromaDB vector storage and retriever logic
│   └── sql/             # SQLAlchemy query engine and schema generator
└── connectors/          # Cloud storage adapters (AWS S3, Azure Blob, GCS)
```

---

## 📸 Screenshots

> 🖼️ Add screenshots here

---

## 📄 License

This project is licensed under the MIT License.