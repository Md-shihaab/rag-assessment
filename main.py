from dotenv import load_dotenv
load_dotenv()

import os
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import groq
import json

# ================= CONFIG =================
DOCS_FOLDER = "docs/"
PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# ================= EMBEDDINGS =================
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "mps"}
)

# ================= BUILD / LOAD VECTOR DB =================
def load_vectorstore():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("Loading existing Chroma DB...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

    print("Building new Chroma DB...")
    loader = DirectoryLoader(
        DOCS_FOLDER,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        recursive=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF pages.")

    for d in documents:
        d.metadata["source"] = os.path.basename(d.metadata.get("source", "unknown"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250

    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return db

# ================= PROMPT =================
PROMPT = PromptTemplate.from_template(
    """You are an expert on sustainability in building & construction.

Rules:
- Answer using ONLY the information in the context.
- You MAY summarize and combine multiple statements from the context.
- Do NOT add facts that are not supported by the context.
- If only part can be answered, answer that part and put the rest in "missing".
- If nothing relevant is found, answer must be exactly:
  "Insufficient information in provided documents".

Context:
{context}

Question:
{question}

Return ONLY valid JSON:
{{
  "answer": "short summary grounded in the context",
  "missing": ["what the documents do not provide (if anything)"],
  "sources": ["file1.pdf", "file2.pdf"],
  "confidence": "high|medium|low"
}}
"""

)

# ================= JSON VALIDATION =================
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

def validate_json_output(output: str) -> dict:
    """
    Parse and validate the model output.
    Raises ValueError if invalid.
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON output must be an object (dictionary).")

    # Required keys
    for key in ("answer", "missing", "sources", "confidence"):

        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    # Types
    if not isinstance(data["answer"], str):
        raise ValueError("answer must be a string")
    if not isinstance(data["sources"], list) or not all(isinstance(s, str) for s in data["sources"]):
        raise ValueError("sources must be a list of strings")
    if not isinstance(data["confidence"], str) or data["confidence"] not in ALLOWED_CONFIDENCE:
        raise ValueError("confidence must be one of: high, medium, low")
    if not isinstance(data["missing"], list) or not all(isinstance(x, str) for x in data["missing"]):
        raise ValueError("missing must be a list of strings")


    return data


# ================= LLM =================

llm = ChatGroq(
    model_name=LLM_MODEL,
    temperature=0.1,
    model_kwargs={"max_completion_tokens": 600}
)

# ================= RAG PIPELINE (NO langchain.chains) =================
def format_docs(docs):
    return "\n\n".join(
        f"SOURCE: {d.metadata.get('source')} \n{d.page_content}"
        for d in docs
    )
@retry(
    retry=retry_if_exception_type((groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
)

def query_rag(question, retriever):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    rag_chain = (
        {
            "context": lambda _: context,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    parsed = validate_json_output(result)
    parsed["sources"] = sorted(set(parsed["sources"]))
    
    if parsed["answer"].strip() == "Insufficient information in provided documents":
        parsed["confidence"] = "low"
    elif parsed["missing"]:
        parsed["confidence"] = "medium"




    return {
        "answer": parsed,     
        "documents": docs
    }

# ================= SELF-CHECK =================
CHECK_PROMPT = PromptTemplate.from_template(
    """You are checking grounding of a SUMMARY.

Return Yes if:
- The answer is a fair summary of the context (it may paraphrase),
- and it does not introduce new factual details not found in the context.

Return No if:
- The answer adds ANY new requirement, threshold, date, scope rule, or claim not present in the context.

Return exactly:
Yes - <short reason>
or
No - <short reason>

Answer JSON:
{answer_json}

Context:
{context}
"""
)



def self_check(question, answer_dict, docs):
    context = format_docs(docs)
    check_chain = CHECK_PROMPT | llm | StrOutputParser()
    return check_chain.invoke({
        "question": question,
        "answer_json": json.dumps(answer_dict, ensure_ascii=False),
        "context": context
    })

def enforce_grounding(parsed_answer: dict, check_result: str) -> dict:
    """
    If self-check says 'No', override to safe abstention.
    """
    if check_result.strip().lower().startswith("no"):
        # Keep sources but dedupe
        sources = []
        for s in parsed_answer.get("sources", []):
            if s not in sources:
                sources.append(s)

        return {
            "answer": "Insufficient information in provided documents",
            "sources": sources,
            "confidence": "low"
        }
    return parsed_answer

# ================= RUN DEMOS =================
demo_queries = [
    "What are the key CSRD reporting requirements for a medium-sized construction company in Germany?",
    "Suggest one regenerative material innovation to reduce embodied carbon in building renovations, including any relevant regulatory context from GEG or EU rules.",
    "What are the main GEG compliance risks when using heat pumps in a new office building in Berlin?"
]

if __name__ == "__main__":
    REBUILD = os.getenv("REBUILD_DB", "0") == "1"
    if REBUILD and os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 40})



    print("\n=== Running Demo Queries ===\n")
    for q in demo_queries:
        print("Query:", q)
        result = query_rag(q, retriever)

        
        check = self_check(q, result["answer"], result["documents"])


        print("Answer JSON:", json.dumps(result["answer"], indent=2, ensure_ascii=False))
        print("Sources:", [d.metadata["source"] for d in result["documents"]])
        
        print("Self-check:", check)
        
