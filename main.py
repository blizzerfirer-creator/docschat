from __future__ import annotations

"""
DocsChat — AI-powered document Q&A
Upload PDFs → Ask questions → Get answers with sources

Stack: FastAPI + TF-IDF search + Abacus.AI RouteLLM (multi-model)
"""

import json
import math
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from PyPDF2 import PdfReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ABACUS_KEY = os.environ.get("ABACUS_API_KEY", "")
ROUTELLM_URL = "https://routellm.abacus.ai/v1"
LLM_MODEL = "route-llm"

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DocsChat", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n\n"
    return text.strip()


def chunk_text(text: str) -> list[dict]:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    chunks = []
    paragraphs = text.split('\n\n')
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < CHUNK_SIZE:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(para) > CHUNK_SIZE:
                words = para.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 < CHUNK_SIZE:
                        current += word + " "
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        current = word + " "
                if current.strip():
                    current += "\n\n"
            else:
                current = para + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return [{"id": f"chunk_{i}", "text": c, "index": i} for i, c in enumerate(chunks)]


# ---------------------------------------------------------------------------
# TF-IDF Search (no external dependencies needed)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w{2,}\b', text.lower())


def build_index(chunks: list[dict]) -> dict:
    """Build a simple TF-IDF index from chunks."""
    docs = [tokenize(c["text"]) for c in chunks]
    n = len(docs)

    # Document frequency
    df = Counter()
    for doc in docs:
        df.update(set(doc))

    # IDF
    idf = {term: math.log(n / (1 + freq)) for term, freq in df.items()}

    # TF-IDF vectors per doc
    vectors = []
    for doc in docs:
        tf = Counter(doc)
        total = len(doc) or 1
        vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
        vectors.append(vec)

    return {"vectors": vectors, "idf": idf}


def search(query: str, index: dict, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Search chunks using TF-IDF cosine similarity."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    idf = index["idf"]
    vectors = index["vectors"]

    # Query vector
    qtf = Counter(query_tokens)
    total = len(query_tokens)
    qvec = {term: (count / total) * idf.get(term, 0) for term, count in qtf.items()}

    # Cosine similarity
    scores = []
    for i, dvec in enumerate(vectors):
        dot = sum(qvec.get(t, 0) * dvec.get(t, 0) for t in qvec)
        mag_q = math.sqrt(sum(v ** 2 for v in qvec.values())) or 1
        mag_d = math.sqrt(sum(v ** 2 for v in dvec.values())) or 1
        sim = dot / (mag_q * mag_d)
        if sim > 0:
            scores.append((sim, i))

    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:top_k]]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def ask_llm(question: str, context: str, history: list[dict] = None) -> str:
    system = """You are a helpful document assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information to answer, say so clearly.
Always cite which part of the document your answer comes from.
Be concise but thorough. Use the same language as the user's question."""

    messages = [{"role": "system", "content": system}]

    if history:
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})

    user_msg = f"""Context from uploaded documents:
---
{context}
---

Question: {question}"""

    messages.append({"role": "user", "content": user_msg})

    resp = httpx.post(
        f"{ROUTELLM_URL}/chat/completions",
        headers={"Authorization": f"Bearer {ABACUS_KEY}", "Content-Type": "application/json"},
        json={"model": LLM_MODEL, "messages": messages, "max_tokens": 1500, "temperature": 0.1},
        timeout=30,
    )

    if resp.status_code != 200:
        raise HTTPException(500, f"LLM error: {resp.text[:200]}")

    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")

    file_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    content = await file.read()
    file_path.write_bytes(content)

    text = extract_text_from_pdf(file_path)
    if not text.strip():
        raise HTTPException(400, "Could not extract text from PDF")

    chunks = chunk_text(text)
    index = build_index(chunks)
    pages = len(PdfReader(str(file_path)).pages)

    session_id = file_id
    sessions[session_id] = {
        "file_id": file_id,
        "filename": file.filename,
        "chunks": chunks,
        "index": index,
        "pages": pages,
        "chars": len(text),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }

    # Limpar arquivo depois de processar
    file_path.unlink(missing_ok=True)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "pages": pages,
        "chunks": len(chunks),
        "chars": len(text),
        "message": f"Document processed: {len(chunks)} chunks from {pages} pages",
    }


@app.post("/api/ask")
async def ask_question(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    question = body.get("question", "").strip()

    if not session_id or session_id not in sessions:
        raise HTTPException(404, "Session not found. Upload a document first.")
    if not question:
        raise HTTPException(400, "Question is required")

    session = sessions[session_id]
    results = search(question, session["index"], session["chunks"])

    context = "\n\n---\n\n".join([r["text"] for r in results])

    if not context.strip():
        return {"answer": "No relevant information found in the document.", "sources": []}

    answer = ask_llm(question, context, session["history"])

    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "sources": [{"text": r["text"][:150] + "..." if len(r["text"]) > 150 else r["text"], "index": r["index"]} for r in results],
        "model": LLM_MODEL,
    }


@app.get("/api/sessions")
async def list_sessions():
    return [
        {"session_id": sid, "filename": s["filename"], "pages": s["pages"], "chunks": len(s["chunks"]), "created_at": s["created_at"]}
        for sid, s in sessions.items()
    ]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    sessions.pop(session_id)
    return {"message": "Session deleted"}


# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if INDEX_HTML.exists():
        return INDEX_HTML.read_text()
    return "<h1>DocsChat</h1><p>static/index.html not found</p>"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
