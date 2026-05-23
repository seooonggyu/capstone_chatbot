import os, re, hashlib, unicodedata
from typing import List
from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader, TextLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from core.config import ENABLE_BM25, K_SPARSE, DEFAULT_SPACE_ID
from kiwipiepy import Kiwi
kiwi = Kiwi()

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    BM25Retriever = None

def preprocess_text(text: str) -> str:
    if not text: return ""
    text = text.replace('\x00', '')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in (meta or {}).items():
        if v is None: continue
        if isinstance(v, (str, int, float, bool)): clean[k] = v
        else: clean[k] = str(v)
    return clean

def custom_tokenize(text: str) -> List[str]:
    """Kiwi를 이용해 의미 있는 명사와 동사 어근만 추출하는 커스텀 토크나이저"""
    if not text:
        return []
    return [t.form for t in kiwi.tokenize(text) if t.tag.startswith('N') or t.tag.startswith('V')]

# def split_semantic_then_fallback(docs: List[Document]) -> List[Document]:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80, separators=["\n\n", "\n", " ", ""])
#     return splitter.split_documents(docs)
def split_semantic_then_fallback(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(docs)

def stable_doc_id(source_uri: str, space_id: str) -> str:
    return hashlib.sha1(f"{space_id}::{source_uri}".encode("utf-8")).hexdigest()

def supported_ext(name: str) -> bool:
    nl = (name or "").lower()
    return nl.endswith((".pdf", ".txt", ".md", ".docx", ".doc"))

def load_and_chunk_from_path(path: str, source_label: str, space_id: str, doc_id: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    docs = []
    try:
        if ext == ".pdf": docs = PyMuPDFLoader(path).load()
        elif ext in [".txt", ".md"]:
            raw = TextLoader(path, encoding="utf-8").load()[0].page_content
            docs = [Document(page_content=raw, metadata={"source": source_label})]
        elif ext in [".docx", ".doc"]: docs = Docx2txtLoader(path).load()
        else: return []
    except Exception as e:
        print(f"문서 파싱 에러 ({ext}): {e}")
        return []

    for d in docs:
        d.page_content = preprocess_text(d.page_content)
        d.metadata = d.metadata or {}
        d.metadata["source"] = source_label

    chunks = split_semantic_then_fallback(docs)
    cleaned = []
    MIN_CHUNK_LEN = 10

    for i, d in enumerate(chunks):
        if not d.page_content or not d.page_content.strip(): continue
        if len(d.page_content.strip()) < MIN_CHUNK_LEN: continue
        d.metadata = d.metadata or {}
        d.metadata.update({"doc_id": doc_id, "chunk_index": i, "source": source_label, "space_id": space_id})
        d.metadata = sanitize_metadata(d.metadata)
        cleaned.append(d)
    return filter_complex_metadata(cleaned)

# async def rebuild_bm25(app: FastAPI, space_id: str = DEFAULT_SPACE_ID) -> None:
#     if not ENABLE_BM25 or BM25Retriever is None:
#         app.state.bm25[space_id] = None
#         return
#     raw = app.state.vectordb._collection.get(where={"space_id": space_id}, include=["documents", "metadatas"])
#     docs = [Document(page_content=t, metadata=(m or {})) for t, m in zip(raw.get("documents", []), raw.get("metadatas", [])) if t and t.strip()]
#     if docs:
#         bm25 = BM25Retriever.from_documents(docs)
#         bm25.k = K_SPARSE
#         app.state.bm25[space_id] = bm25
#     else:
#         app.state.bm25[space_id] = None
async def rebuild_bm25(app: FastAPI, space_id: str = DEFAULT_SPACE_ID) -> None:
    if not ENABLE_BM25 or BM25Retriever is None:
        app.state.bm25[space_id] = None
        return
    raw = app.state.vectordb._collection.get(where={"space_id": space_id}, include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=(m or {})) for t, m in zip(raw.get("documents", []), raw.get("metadatas", [])) if t and t.strip()]
    if docs:
        bm25 = BM25Retriever.from_documents(docs, preprocess_func=custom_tokenize)
        bm25.k = K_SPARSE
        app.state.bm25[space_id] = bm25
    else:
        app.state.bm25[space_id] = None