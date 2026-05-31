# 🧠 Smart Local RAG API (Document Q&A System)

A privacy-first, local Retrieval-Augmented Generation (RAG) API built to ingest PDF documents, chunk them, and answer user questions using semantic search and local LLMs. 

Unlike cloud-based RAG systems, this architecture runs **100% on-premise (Local)** using Ollama, ensuring that sensitive company data never leaves the local environment.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)
![Ollama](https://img.shields.io/badge/LLM-Ollama-purple.svg)

## 🌟 Overview & Architecture
This system bridges the gap between proprietary company documents and Large Language Models. It follows a strict 5-step pipeline:
1. **Ingestion:** Extracts text from uploaded PDFs using `pypdf`.
2. **Chunking:** Splits large texts into semantic chunks using `LangChain Text Splitters` to respect LLM context windows.
3. **Embedding:** Converts text chunks into vector representations using `nomic-embed-text` via local Ollama.
4. **Storage:** Stores vectors and metadata in a local `ChromaDB` instance.
5. **Generation:** Retrieves relevant context based on user queries and generates answers using a local LLM (e.g., `qwen2.5:7b` or `llama3.2`) with strict guardrails to prevent hallucinations.

## 🛠️ Tech Stack
- **Backend:** FastAPI, Uvicorn, Pydantic
- **AI/GenAI:** Ollama (Local LLM & Embeddings), ChromaDB (Vector Database)
- **Document Processing:** PyPDF, LangChain
- **Language:** Python 3.10+

## 🚀 Prerequisites
Before running the API, you must have [Ollama](https://ollama.com/download) installed and running on your machine.

Pull the required models via your terminal:
```bash
# For Embeddings
ollama pull nomic-embed-text

# For Generation (Choose one)
ollama pull qwen2.5:7b
# or
ollama pull llama3.2