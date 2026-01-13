Echocene RAG Assessment – Sustainability & Construction 🌱🏗️

Candidate: Mohammed Shihaab R
Date: January 2026

This repository contains a demo Retrieval-Augmented Generation (RAG) system built as part of the Echocene technical assessment. The system answers sustainability and regulatory questions in the building and construction sector using only public documents and free tools, with a strong focus on accuracy, grounding, and transparency.

--------------------------------------------------

Project Objective 🎯

The goal of this project is to demonstrate the ability to:
- Build a simple and reliable RAG pipeline
- Integrate a free Large Language Model (Groq)
- Retrieve and ground answers in real regulatory documents
- Reduce hallucinations through defensive design
- Return structured, verifiable outputs

This is a demo and assessment project, not production software.

--------------------------------------------------

Tech Stack 🧰

- Python
- LangChain (open-source)
- Groq API (free tier)
- Chroma (local vector database)
- Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2)

No paid APIs or services are used.

--------------------------------------------------

Knowledge Base 📚

The system uses publicly available PDF documents related to:
- Corporate Sustainability Reporting Directive (CSRD)
- EU sustainability and taxonomy regulations
- German Buildings Energy Act (GEG) summaries (English)
- Sustainable construction and building materials

All documents are stored locally in the docs/ folder and embedded into a local Chroma database.

--------------------------------------------------

How the System Works ⚙️

1. PDF documents are loaded from the docs/ folder
2. Documents are split into overlapping text chunks
3. Each chunk is embedded using a Hugging Face model
4. Embeddings are stored locally in Chroma
5. Relevant chunks are retrieved for each query
6. A Groq LLM generates answers using only retrieved context
7. Answers are returned in structured JSON format
8. A post-generation self-check verifies grounding

--------------------------------------------------

Output Format 🧾

Each answer is returned as structured JSON with the following fields:
- answer: grounded summary based on retrieved documents
- missing: information not explicitly available in the documents
- sources: document file names used
- confidence: high, medium, or low

If no relevant information is found, the system returns:
Insufficient information in provided documents

Partial answers are allowed and clearly marked.

--------------------------------------------------

Hallucination Mitigation 🛑

The system includes lightweight safeguards:
- Context-only prompting (no external knowledge)
- Explicit instructions not to guess or infer
- Structured JSON validation
- Clear handling of missing information
- Post-generation grounding self-check

Accuracy is prioritized over completeness.

--------------------------------------------------

Setup & Run ▶️

1. Install dependencies using pip
2. Create a .env file containing your Groq API key
3. Run the main script to build the database and execute demo queries

The vector database is stored locally and can be rebuilt when needed.

--------------------------------------------------

Environment Variables 🔐

- GROQ_API_KEY: your Groq API key
- REBUILD_DB: set to 1 to force rebuilding the vector database

The .env file is excluded from version control via .gitignore.

--------------------------------------------------

Demo Queries 🔍

The system runs the following queries exactly as specified in the assessment:
1. What are the key CSRD reporting requirements for a medium-sized construction company in Germany?
2. Suggest one regenerative material innovation to reduce embodied carbon in building renovations.
3. What are the main GEG compliance risks when using heat pumps in a new office building in Berlin?

For each query, the system prints:
- Retrieved document sources
- Final JSON answer
- Grounding self-check result

--------------------------------------------------

Final Note ✨

Regulatory documents rarely contain perfect, ready-made answers. This system reflects real-world regulatory intelligence challenges by allowing grounded summaries while preventing unsupported claims.
