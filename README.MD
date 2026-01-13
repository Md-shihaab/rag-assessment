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

--------------------------------------------------
Demo Outputs 🧪

Below are the full outputs produced by running:
python main.py

Query 1
Question:
What are the key CSRD reporting requirements for a medium-sized construction company in Germany?

Answer JSON:
{
  "answer": "A medium-sized construction company in Germany must prepare, assure, and publish a sustainability report under the Corporate Sustainability Reporting Directive (CSRD) if it meets the size criteria and has transferable securities admitted to trading on an EU regulated market. The report must be published in a data extractable format together with relevant accompanying metadata. The company must also apply 'reasonable effort' to collect value chain information from actors in its value chain.",
  "missing": [
    "specific details on the format of the sustainability report",
    "exemptions for specific industries or sectors"
  ],
  "sources": [
    "240807-faqs-corporate-sustainability-reporting_en.pdf"
  ],


  "confidence": "medium"
}

Retrieved sources (chunks):
['240807-faqs-corporate-sustainability-reporting_en.pdf']

Self-check:
Yes - The answer is a fair summary of the context and does not introduce new factual details not found in the context.

--------------------------------------------------

Query 2
Question:
Suggest one regenerative material innovation to reduce embodied carbon in building renovations, including any relevant regulatory context from GEG or EU rules.

Answer JSON:
{
  "answer": "Using bio-based materials such as timber or bamboo to replace concrete-based exterior wall systems can greatly reduce embodied carbon and operational emissions in building renovations.",
  "missing": [
    "Specific regulatory context from GEG or EU rules that directly address regenerative material innovations in building renovations"
  ],
  "sources": [
    "Building_materials_climate.pdf",
    "global_status_report_buildings_construction_2023.pdf"
  ],


  "confidence": "medium"
}

Retrieved sources (chunks):
['Building_materials_climate.pdf','global_status_report_buildings_construction_2023.pdf','global_status_report_buildings_construction_2023.pdf']

Self-check:
Yes - The answer paraphrases the context and does not introduce new factual details not found in the context.

--------------------------------------------------

Query 3
Question:
What are the main GEG compliance risks when using heat pumps in a new office building in Berlin?

Answer JSON:
{
  "answer": "The main GEG compliance risks when using heat pumps in a new office building in Berlin are related to the minimum share of renewable energy required for new heating systems, which is at least 65% as per the Buildings Energy Act (GEG). Additionally, landlords can pass on up to 10% of the costs for climate-friendly heating to tenants, but the monthly basic rent can be increased by a maximum of 50 cents per square meter.",
  "missing": [
    "specific deadlines and transitional provisions for new office buildings in Berlin",
    "information on how to calculate the minimum share of renewable energy for the new office building"
  ],
  "sources": [
    "Efficient buildings _ BMWE.pdf",
    "faktenblatt-geg-gebaeudeenergiegesetz-en.pdf"
  ],



  "confidence": "medium"
}

Retrieved sources (chunks):
['Efficient buildings _ BMWE.pdf', 'faktenblatt-geg-gebaeudeenergiegesetz-en.pdf', 'climate-action-programme-2023.pdf', 'global_status_report_buildings_construction_2023.pdf', 'climate-action-programme-2023.pdf']

Self-check:
Yes - The answer is a fair summary of the context and does not introduce new factual details not found in the context.




--------------------------------------------------

