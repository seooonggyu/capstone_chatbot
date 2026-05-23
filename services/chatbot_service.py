import time, math, os, asyncio, traceback
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from core.config import *
from schemas.dto import ChatRequest, ChatResponse, SourceInfo
from services.document_service import load_and_chunk_from_path, stable_doc_id, rebuild_bm25
from sentence_transformers import CrossEncoder
reranker_model = CrossEncoder("Dongjin-kr/ko-reranker", max_length=512, device="cuda")

def is_korean(text: str) -> bool:
    return any(("\uAC00" <= c <= "\uD7A3") or ("\u1100" <= c <= "\u11FF") or ("\u3130" <= c <= "\u318F") for c in (text or ""))

def build_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts, total = [], 0
    for d in docs:
        text = (d.page_content or "").strip()
        if not text: continue

        source_name = (d.metadata or {}).get("source", "알 수 없는 문서")
        labeled_text = f"[문서명: {source_name}]\n{text}"

        if total + len(labeled_text) > max_chars:
            break
        parts.append(labeled_text)
        total += len(labeled_text)
    return "\n\n---\n\n".join(parts)

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)

def rerank_with_cross_encoder(query: str, docs: List[Document], top_k: int) -> List[Document]:
    if not docs: return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:top_k] if score > 0.0]

def reciprocal_rank_fusion(dense_docs: List[Document], sparse_docs: List[Document], k: int = 60) -> List[Document]:
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        doc_key = str(doc.metadata.get("doc_id", "")) + ":" + str(doc.metadata.get("chunk_index", ""))
        rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (rank + k)
        doc_map[doc_key] = doc

    if sparse_docs:
        for rank, doc in enumerate(sparse_docs):
            doc_key = str(doc.metadata.get("doc_id", "")) + ":" + str(doc.metadata.get("chunk_index", ""))
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (rank + k)
            doc_map[doc_key] = doc

    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]

async def process_chat(req: ChatRequest, app: FastAPI) -> ChatResponse:
    start = time.time()
    q = (req.question or "").strip()
    if not q: raise HTTPException(status_code=400, detail="question is required")

    space_id = (req.space_id or DEFAULT_SPACE_ID).strip() or DEFAULT_SPACE_ID
    vectordb = app.state.vectordb
    bm25 = getattr(app.state, "bm25", {}).get(space_id)
    answer_language = "Korean" if is_korean(q) else "English"
    search_query = q

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": K_DENSE,
            "fetch_k": FETCH_K,
            "lambda_mult": LAMBDA_MULT,
            "filter": {"space_id": space_id}
        }
    )
    dense_candidates = list(await asyncio.to_thread(retriever.invoke, search_query))

    sparse_candidates = []
    if ENABLE_BM25 and bm25 is not None:
        try:
            bm25.k = K_SPARSE
            raw_sparse = await asyncio.to_thread(bm25.invoke, search_query)
            sparse_candidates = [d for d in raw_sparse if (d.metadata or {}).get("space_id") == space_id]
        except Exception as e:
            print(f"BM25 Search Error: {e}")

    rrf_candidates = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60)
    top_candidates = rrf_candidates[:15]
    final_docs = await asyncio.to_thread(rerank_with_cross_encoder, q, top_candidates, K_FINAL)

    if not final_docs:
        answer = "관련 자료에서 답을 찾지 못했습니다." if answer_language == "Korean" else "I don't have information about that."
        return ChatResponse(answer=answer, sources=[])

    context = build_context(final_docs, max_chars=MAX_CONTEXT_CHARS)
    chain = BASE_PROMPT | llm | StrOutputParser()
    answer = (await chain.ainvoke({"context": context, "question": q, "answer_language": answer_language}) or "").strip()

    sources = []

    if "정보없음" in answer.replace(" ", ""):
        return ChatResponse(
            answer="관련 자료에서 답을 찾지 못했습니다. 다른 키워드로 질문해 주시거나 추가 자료를 업로드해 주세요.",
            sources=[]
        )

    no_answer_keywords = [
        "관련 자료에서 답을 찾지 못했",
        "정보를 가지고 있지 않습니다",
        "정보가 없습니다",
        "정보는 없습니다",
        "찾을 수 없습니다",
        "찾지 못했습니다",
        "알 수 없습니다",
        "언급되어 있지 않습니다",
        "내용이 없습니다",
        "답변할 수 없습니다",
        "제공되지 않았습니다",
        "I don't have information about",
        "not mentioned in the",
        "not available in the provided",
    ]

    if not any(keyword in answer for keyword in no_answer_keywords):
        seen_sources = set()
        for d in final_docs:
            meta = d.metadata or {}
            source_name = meta.get("source", "unknown")
            page_num = meta.get("page")
            page_val = int(page_num) + 1 if page_num is not None else None

            uniq_key = f"{source_name}_{page_val}"
            if uniq_key not in seen_sources:
                sources.append(SourceInfo(source=source_name, page=page_val))
                seen_sources.add(uniq_key)

    print(f"⏱️ 챗봇 응답 소요 시간: {time.time() - start:.2f}초")
    return ChatResponse(answer=answer, sources=sources)

async def process_ingest(file_path: str, space_id: str, app: FastAPI, user_id: str = "Unknown") -> dict:
    try:
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}", flush=True)
            return {"status": "error", "message": f"File not found: {file_path}"}

        vectordb = app.state.vectordb
        uri, source_label = f"file://{os.path.abspath(file_path)}", os.path.basename(file_path)
        doc_id = stable_doc_id(uri, space_id)

        chunks = await asyncio.to_thread(load_and_chunk_from_path, file_path, source_label, space_id, doc_id)

        if not chunks: return {"status": "skipped", "message": "지원하지 않는 확장자이거나 추출할 텍스트가 없습니다."}

        async with app.state.write_lock:
            if old_ids := vectordb._collection.get(where={"doc_id": doc_id}).get("ids", []): vectordb._collection.delete(ids=old_ids)
            batch_size = 5
            skipped = 0
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]

                try:
                    await asyncio.to_thread(vectordb.add_documents, batch_chunks)
                except Exception:
                    for single_chunk in batch_chunks:
                        try:
                            await asyncio.to_thread(vectordb.add_documents, [single_chunk])
                        except Exception as chunk_err:
                            skipped += 1
                            print(f"⚠️ [SKIP] 청크 임베딩 실패 (건너뜀): {chunk_err}", flush=True)

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_chunk = min(i + batch_size, len(chunks))
                print(f"[{now}] [User: {user_id} | Space: {space_id}] 임베딩 진행 중 ...({current_chunk}/{len(chunks)})", flush=True)
                await asyncio.sleep(0.1)

            if skipped:
                print(f"⚠️ 총 {skipped}개 청크 임베딩 실패로 제외됨", flush=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] [User: {user_id} | Space: {space_id}] 임베딩 완료", flush=True)
        async with app.state.rebuild_lock: await rebuild_bm25(app, space_id=space_id)
        return {"status": "success", "message": f"성공적으로 {len(chunks)}개의 청크를 DB에 추가했습니다.", "space_id": space_id}
    except Exception as e:
        print(f"[ERROR] 백그라운드 에러 발생")
        print(f"[ERROR] 에러 원인: {str(e)}", flush=True)
        traceback.print_exc()
        return {"status": "error", "message": str(e)}