from __future__ import annotations

"""
DocsChat — AI-powered document Q&A
Upload PDFs → Ask questions → Get answers with sources

Stack: FastAPI + ChromaDB + Abacus.AI RouteLLM (multi-model)
"""

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import httpx
from PyPDF2 import PdfReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ABACUS_KEY = os.environ.get("ABACUS_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ROUTELLM_URL = "https://routellm.abacus.ai/v1"

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 800  # chars per chunk
CHUNK_OVERLAP = 150

# LLM config
LLM_MODEL = "route-llm"  # auto-routes to best model

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DocsChat", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ChromaDB — usa default embedding function (all-MiniLM-L6-v2 local)
chroma = chromadb.Client()

# Pré-carregar modelo de embedding na inicialização
from chromadb.utils import embedding_functions
embed_fn = embedding_functions.DefaultEmbeddingFunction()

# Store sessions
sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: Path) -> str:
    """Extrai texto de um PDF."""
    reader = PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Divide texto em chunks com overlap."""
    # Limpar texto
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    chunks = []
    # Tentar dividir por parágrafos primeiro
    paragraphs = text.split('\n\n')
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            # Se o parágrafo é maior que chunk_size, dividir por frases
            if len(para) > chunk_size:
                words = para.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 < chunk_size:
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

    # Adicionar metadata
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "id": f"chunk_{i}",
            "text": chunk,
            "index": i,
            "chars": len(chunk),
        })

    return result




# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def ask_llm(question: str, context: str, history: list[dict] = None) -> str:
    """Pergunta ao LLM com contexto dos documentos."""
    system = """You are a helpful document assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information to answer, say so clearly.
Always cite which part of the document your answer comes from.
Be concise but thorough. Use the same language as the user's question."""

    messages = [{"role": "system", "content": system}]

    # Histórico de conversa
    if history:
        for h in history[-6:]:  # últimas 3 trocas
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
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.1,
        },
        timeout=30,
    )

    if resp.status_code != 200:
        raise HTTPException(500, f"LLM error: {resp.text[:200]}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload e processa um PDF."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")

    # Salvar arquivo
    file_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    content = await file.read()
    file_path.write_bytes(content)

    # Extrair texto
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        raise HTTPException(400, "Could not extract text from PDF")

    # Chunkar
    chunks = chunk_text(text)

    # Collection no ChromaDB (usa embedding local automático)
    collection_name = f"doc_{file_id}"
    collection = chroma.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Adicionar chunks (ChromaDB gera embeddings automaticamente)
    batch_size = 40
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[f"{file_id}_{c['id']}" for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"index": c["index"], "file": file.filename} for c in batch]
        )

    # Criar sessão
    session_id = file_id
    sessions[session_id] = {
        "file_id": file_id,
        "filename": file.filename,
        "collection": collection_name,
        "chunks": len(chunks),
        "chars": len(text),
        "pages": len(PdfReader(str(file_path)).pages),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "pages": sessions[session_id]["pages"],
        "chunks": len(chunks),
        "chars": len(text),
        "message": f"Document processed: {len(chunks)} chunks from {sessions[session_id]['pages']} pages",
    }


@app.post("/api/ask")
async def ask_question(request: Request):
    """Faz uma pergunta sobre o documento."""
    body = await request.json()
    session_id = body.get("session_id")
    question = body.get("question", "").strip()

    if not session_id or session_id not in sessions:
        raise HTTPException(404, "Session not found. Upload a document first.")
    if not question:
        raise HTTPException(400, "Question is required")

    session = sessions[session_id]
    collection = chroma.get_collection(session["collection"], embedding_function=embed_fn)

    # Buscar chunks relevantes (ChromaDB gera embedding da query automaticamente)
    results = collection.query(
        query_texts=[question],
        n_results=5,
    )

    # Montar contexto
    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n---\n\n".join(docs)

    if not context.strip():
        return {"answer": "No relevant information found in the document.", "sources": []}

    # Perguntar ao LLM
    answer = ask_llm(question, context, session["history"])

    # Salvar no histórico
    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "sources": [{"text": d[:150] + "..." if len(d) > 150 else d, "index": i} for i, d in enumerate(docs)],
        "model": LLM_MODEL,
    }


@app.get("/api/sessions")
async def list_sessions():
    """Lista sessões ativas."""
    return [
        {
            "session_id": sid,
            "filename": s["filename"],
            "pages": s["pages"],
            "chunks": s["chunks"],
            "created_at": s["created_at"],
        }
        for sid, s in sessions.items()
    ]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Remove uma sessão."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions.pop(session_id)
    try:
        chroma.delete_collection(session["collection"])
    except Exception:
        pass

    return {"message": "Session deleted"}


# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
