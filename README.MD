Echocene RAG Assessment – Sustainability & Construction 🌱🏗️

Candidate: Mohammed Shihaab R
Date: January 2026

This project is a demo Retrieval-Augmented Generation (RAG) system built for the Echocene technical assessment. It answers sustainability and regulatory questions in the construction sector using only public documents and free tools, with a strong focus on accuracy and grounding.

What This Project Shows 🎯

This project demonstrates how to:

Build a simple but reliable RAG pipeline

Use a free LLM (Groq) effectively

Ground answers strictly in retrieved documents

Reduce hallucinations with defensive design

Return clear, structured, and verifiable outputs

This is an assessment/demo project, not production software.

Tech Stack 🧰

Python

LangChain (open source)

Groq API (free tier)

Chroma (local vector database)

Hugging Face embeddings (MiniLM)

No paid APIs or services are used.

Knowledge Base 📚

The system uses publicly available PDFs covering:

CSRD (Corporate Sustainability Reporting Directive)

EU sustainability and taxonomy regulations

German Buildings Energy Act (GEG) summaries (English)

Sustainable construction and building materials

All documents are stored locally in the docs/ folder.

How It Works ⚙️

PDFs are loaded and split into text chunks

Chunks are embedded using a Hugging Face model

Embeddings are stored locally in Chroma

Relevant chunks are retrieved per query

Groq generates answers using only retrieved context

Answers are returned as structured JSON

A self-check verifies grounding

Output Format 🧾

Each answer is returned as JSON with:

answer – grounded summary from the documents

missing – what the documents do not explicitly provide

sources – document file names used

confidence – high / medium / low

If nothing relevant is found, the system clearly responds:
“Insufficient information in provided documents”

Partial answers are allowed and clearly marked.

Hallucination Mitigation 🛑

To reduce hallucinations, the system uses:

Context-only prompting

Explicit “no guessing” rules

JSON output validation

Clear abstention when info is missing

Post-generation grounding self-check

Accuracy is always prioritized over completeness.

Setup & Run ▶️

Install dependencies with pip

Create a .env file with your Groq API key

Run the main script

The vector database is stored locally and can be rebuilt if needed.

Environment Variables 🔐

GROQ_API_KEY – your Groq API key

REBUILD_DB – set to 1 to rebuild the vector database

The .env file is ignored by Git.

Demo Queries 🔍

The system runs these queries exactly as required:

CSRD reporting requirements for a medium-sized construction company in Germany

Regenerative material innovation to reduce embodied carbon

GEG compliance risks for heat pumps in a new office building in Berlin

For each query, the system prints:

Retrieved sources

Final JSON answer

Grounding self-check result

Final Note ✨

Regulatory documents rarely contain perfect, ready-made answers. This system reflects real-world constraints by allowing grounded summaries while preventing unsupported claims — a key challenge in regulatory and sustainability intelligence.
