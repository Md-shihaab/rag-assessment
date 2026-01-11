```bash
# Echocene RAG Assessment – Sustainability & Construction

**Candidate:** Mohammed Shihaab R  
**Date:** January 2026  

This repository contains a Python-based Retrieval-Augmented Generation (RAG) system built as part of the Echocene technical assessment. The system provides grounded answers on sustainability and regulatory topics in the building and construction sector, with a focus on CSRD and GEG.

---

## Objective

The goal of this project is to demonstrate the ability to:

- Build a reliable RAG pipeline
- Integrate a free LLM (Groq)
- Generate factual, document-grounded answers
- Reduce hallucinations through defensive design
- Return structured and verifiable outputs

---

## Tech Stack

- Python  
- LangChain (open-source)  
- Groq API (free tier)  
- Chroma (local vector database)  
- Hugging Face embeddings  
  - sentence-transformers/all-MiniLM-L6-v2  

No paid tools or APIs are used.

---

## Knowledge Base

The knowledge base consists of publicly available PDF documents related to:

- Corporate Sustainability Reporting Directive (CSRD)
- EU sustainability and taxonomy regulations
- German Buildings Energy Act (GEG) summaries (English)
- Sustainable construction and building materials

All documents are stored locally in the `docs/` folder and embedded into a local Chroma database.

---

## How the System Works

1. PDF documents are loaded and split into chunks
2. Chunks are embedded using a Hugging Face model
3. Embeddings are stored locally in Chroma
4. Relevant chunks are retrieved for each query
5. A Groq LLM generates answers using only retrieved context
6. Answers are returned in strict JSON format
7. A post-generation self-check verifies grounding

---

## Output Format

Each answer is returned as JSON:

```json
{
  "answer": "Grounded summary based on retrieved documents",
  "missing": ["Information not explicitly available in documents"],
  "sources": ["document_name.pdf"],
  "confidence": "high | medium | low"
}





# Echocene RAG Assessment – Sustainability & Construction

Candidate: Mohammed Shihaab R  
Date: January 2026  

This repository contains a Python-based Retrieval-Augmented Generation (RAG) system built as part of the Echocene technical assessment. The system answers sustainability and regulatory questions in the building and construction sector, with a focus on CSRD and GEG.

---

## Objective

The goal of this project is to demonstrate the ability to:

- Build a reliable RAG pipeline
- Integrate a free LLM (Groq)
- Retrieve and ground answers in public documents
- Reduce hallucinations using simple defensive techniques
- Return structured, verifiable outputs

---

## Tech Stack

- Python
- LangChain (open-source)
- Groq API (free tier)
- Chroma (local vector database)
- Hugging Face embeddings  
  sentence-transformers/all-MiniLM-L6-v2

No paid tools or APIs are used.

---

## Knowledge Base

The system uses publicly available PDF documents related to:

- Corporate Sustainability Reporting Directive (CSRD)
- EU sustainability and taxonomy regulations
- German Buildings Energy Act (GEG) summaries (English)
- Sustainable construction and building materials

All documents are stored locally in the `docs/` folder and embedded into a local Chroma database.

---

## How the System Works

1. PDF documents are loaded and split into text chunks
2. Chunks are embedded using a Hugging Face model
3. Embeddings are stored locally in Chroma
4. Relevant chunks are retrieved for each query
5. A Groq LLM generates answers using only retrieved context
6. Answers are returned in structured JSON
7. A post-generation self-check verifies grounding

---

## Output Format

Each answer is returned as JSON:

{
  "answer": "Grounded summary based on retrieved documents",
  "missing": ["Information not explicitly available in documents"],
  "sources": ["document_name.pdf"],
  "confidence": "high | medium | low"
}

- If no relevant information is found, the answer is:
  "Insufficient information in provided documents"
- Partial answers are allowed and clearly marked.

---

## Hallucination Mitigation

The system includes basic safeguards:

- Context-only prompting
- Structured JSON validation
- Explicit abstention when information is missing
- Post-generation grounding self-check

The design prioritizes accuracy over completeness.

---

## Setup

Install dependencies:

pip install -U langchain langchain-core langchain-community langchain-text-splitters \
langchain-chroma langchain-huggingface langchain-groq chromadb pypdf \
sentence-transformers tenacity python-dotenv

---

## Environment Variables

Create a `.env` file:

GROQ_API_KEY=your_groq_api_key  
REBUILD_DB=1

---

## Run

python main.py

---

## Demo Queries

The following queries are executed exactly as required by the assignment:

1. What are the key CSRD reporting requirements for a medium-sized construction company in Germany?
2. Suggest one regenerative material innovation to reduce embodied carbon in building renovations.
3. What are the main GEG compliance risks when using heat pumps in a new office building in Berlin?

For each query, the system prints:
- Retrieved sources
- Final JSON answer
- Grounding self-check result

---

## Notes

Regulatory documents rarely contain direct, answer-shaped text. This system allows summarization across multiple passages while preventing unsupported claims, reflecting real-world regulatory question-answering constraints.
