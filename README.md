# 인계점 챗봇 (Capstone Chatbot API)

> **RAG(Retrieval-Augmented Generation) 기반의 폐쇄형 문서 챗봇**  
> 사용자가 업로드한 문서만을 유일한 지식 소스로 삼아, 출처와 함께 질의응답을 제공합니다.  
> LLM 자체 서빙(Ollama)으로 외부 API 호출 없이 완전한 온-프레미스 동작을 지향합니다.

---

## 목차

1. [핵심 설계 철학](#1-핵심-설계-철학)
2. [검색 파이프라인 상세](#2-검색-파이프라인-상세)
3. [프롬프트 설계](#3-프롬프트-설계)
4. [문서 전처리 및 청킹 전략](#4-문서-전처리-및-청킹-전략)
5. [스페이스별 데이터 격리](#5-스페이스별-데이터-격리)
6. [기술 스택](#6-기술-스택)
7. [프로젝트 구조](#7-프로젝트-구조)
8. [API 엔드포인트](#8-api-엔드포인트)
9. [실행 방법](#9-실행-방법)

---

## 1. 핵심 설계 철학

이 챗봇은 단순한 LLM 래퍼가 아닙니다. **"업로드된 문서 외의 정보는 절대 사용하지 않는다"** 는 폐쇄형 원칙을 설계 전반에 관철시켰습니다.

- LLM의 사전 학습 지식이 개입하면 인수인계 맥락에서 잘못된 정보가 제공될 수 있습니다.
- 검색된 컨텍스트가 없거나 불충분할 경우, 답변을 지어내는 대신 `정보없음` 키워드를 출력하도록 강제합니다.
- 이를 위해 프롬프트 엔지니어링, 검색 품질, 컨텍스트 구성 세 단계 모두에서 별도의 설계가 이루어졌습니다.

---

## 2. 검색 파이프라인 상세

질문 하나에 대해 총 4단계의 검색 및 정제 과정이 순차 실행됩니다.

```
질문 입력
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1. Dense Retrieval (Chroma + MMR)                 │
│  임베딩 기반 코사인 유사도 검색 (K=20, fetch_k=40)        │
│  + MMR(Maximal Marginal Relevance)로 다양성 확보         │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  STEP 2. Sparse Retrieval (BM25 + Kiwi 형태소 분석)      │
│  Kiwi로 명사/동사 어근만 추출한 커스텀 토크나이저 사용    │
│  키워드 기반 역문서빈도 검색 (K=20)                       │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  STEP 3. RRF (Reciprocal Rank Fusion)                    │
│  Dense + Sparse 결과를 순위 기반으로 합산·병합            │
│  score = Σ 1 / (rank + 60)                              │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│  STEP 4. Cross-Encoder Reranking                         │
│  ko-reranker 모델로 (질문, 청크) 쌍의 관련도 재점수화     │
│  최종 K=12개 청크만 컨텍스트로 LLM에 전달                │
└─────────────────────────┬───────────────────────────────┘
                          │
                    LLM 답변 생성
```

### Dense Retrieval — Chroma + MMR

`mxbai-embed-large` 임베딩 모델로 문서 청크를 벡터화하여 Chroma DB에 저장합니다. 검색 시에는 단순 Top-K가 아닌 **MMR(Maximal Marginal Relevance)** 을 적용합니다. MMR은 질문과의 유사도(relevance)와 이미 선택된 청크와의 다양성(diversity) 사이의 균형을 `lambda_mult=0.65` 파라미터로 조정하여, 중복된 내용의 청크가 컨텍스트를 잠식하는 현상을 방지합니다.

### Sparse Retrieval — BM25 + 한국어 형태소 분석

`rank_bm25` 라이브러리 기반의 BM25 검색기를 운용합니다. 한국어는 어미 변화가 많아 단순 띄어쓰기 토크나이징으로는 검색 품질이 낮아지는 문제가 있습니다. 이를 해결하기 위해 `kiwipiepy` 형태소 분석기로 **명사(N)와 동사 어근(V) 품사만 추출**하는 커스텀 토크나이저를 구현했습니다. Dense 검색이 의미적 유사성에 강점을 갖는 반면, BM25는 고유 명사·전문 용어 등 정확한 키워드 매칭에 보완적으로 작동합니다.

### RRF (Reciprocal Rank Fusion)

두 검색기의 결과를 단순 점수로 합산하면 스케일 차이로 인한 편향이 발생합니다. RRF는 각 결과의 **순위(rank)** 만을 사용해 `1 / (rank + k)` 방식으로 점수를 계산하고 합산합니다. 순위 기반이므로 이질적인 두 검색 방식의 점수를 안전하게 병합할 수 있습니다.

### Cross-Encoder Reranking

RRF 병합 후 후보 청크들을 `Dongjin-kr/ko-reranker` 크로스 인코더 모델로 재점수화합니다. Bi-Encoder(임베딩) 기반 검색은 속도는 빠르지만 질문과 청크를 독립적으로 인코딩하는 한계가 있습니다. 크로스 인코더는 질문과 청크를 **하나의 입력으로 함께 처리**하여 더 정밀한 관련도를 계산합니다. 최종적으로 상위 12개 청크만 LLM에 전달합니다.

---

## 3. 프롬프트 설계

LLM이 학습 지식을 사용하지 못하도록 시스템 프롬프트에 엄격한 규칙을 명시했습니다.

| 규칙 | 내용 |
|------|------|
| **RULE 1 — 완전한 답변** | 컨텍스트로 질문에 완전히 답할 수 있으면 명확하고 구조화된 답변 제공 |
| **RULE 2 — 부분 정보** | 일부만 답할 수 있는 경우, 지원되는 부분만 답하고 나머지는 "정보 없음"임을 명시 |
| **RULE 3 — 정보 없음** | 컨텍스트에 관련 내용이 전혀 없으면 `정보없음` 키워드만 출력 (추측 금지) |
| **RULE 4 — 절대 금지** | 추측, 추론, 내부 지식 활용, 정보 창작 금지 |

컨텍스트 구성 시에는 각 청크 앞에 `[문서명: {source}]` 레이블을 붙여 LLM이 어느 문서에서 해당 내용이 왔는지 인식할 수 있게 하며, 최종 답변에 출처와 페이지 정보를 함께 반환합니다.

---

## 4. 문서 전처리 및 청킹 전략

### 지원 포맷

PDF, TXT, Markdown(`.md`), DOCX

### 전처리 파이프라인

1. **인코딩 정규화**: `unicodedata.normalize('NFKC')`로 한글 자모 분리 등 유니코드 이슈 처리
2. **제어 문자 제거**: 임베딩 모델이 처리하지 못하는 특수 제어 문자 제거
3. **공백 정규화**: 연속 공백/탭을 단일 공백으로 치환

### 청킹 전략

`RecursiveCharacterTextSplitter`를 사용하며, 분리 우선순위는 `문단 → 줄바꿈 → 마침표 → 공백` 순입니다.

| 파라미터 | 값 | 이유 |
|---------|-----|------|
| `chunk_size` | 500 | 크로스 인코더 max_length(512)와 정합성 유지 |
| `chunk_overlap` | 150 | 청크 경계에서 문맥이 잘리는 현상 방지 |

청크마다 `doc_id`(SHA-1 해시), `chunk_index`, `space_id`, `source` 메타데이터를 주입하여 검색 시 식별 및 격리에 활용합니다.

---

## 5. 스페이스별 데이터 격리

단일 Chroma DB에 여러 사용자/팀의 문서를 저장하면서도, 검색 시 타인의 문서가 컨텍스트에 포함되지 않도록 **메타데이터 필터** 기반 격리를 구현했습니다.

- 문서 임베딩 시 `space_id`를 메타데이터로 주입
- Retrieval 단계에서 `{"space_id": req.space_id}` 필터를 적용하여 해당 스페이스의 청크만 검색
- BM25 인덱스도 `app.state.bm25[space_id]` 형태로 스페이스별로 분리 관리

---

## 6. 기술 스택

| 분류 | 기술 |
|------|------|
| **웹 프레임워크** | FastAPI |
| **LLM** | Ollama (`gemma2:9b`) — 로컬 서빙 |
| **임베딩 모델** | `mxbai-embed-large` (Ollama) |
| **벡터 DB** | Chroma |
| **Dense 검색** | LangChain Chroma + MMR |
| **Sparse 검색** | BM25 (`rank_bm25`) + Kiwi 형태소 분석 |
| **결과 병합** | Reciprocal Rank Fusion (RRF) |
| **재순위화** | `Dongjin-kr/ko-reranker` (Cross-Encoder) |
| **LLM 파이프라인** | LangChain |
| **비동기 처리** | FastAPI `BackgroundTasks` |
| **배포 (Colab)** | ngrok + uvicorn |

---

## 7. 프로젝트 구조

```text
.
├── api/
│   └── routes.py               # API 라우팅 — /chatbot, /ingest 엔드포인트
├── core/
│   └── config.py               # 전역 설정 (모델, DB 경로, 검색 파라미터, 프롬프트)
├── schemas/
│   └── dto.py                  # Pydantic 데이터 전송 객체 (요청/응답 스키마)
├── services/
│   ├── chatbot_service.py      # 핵심 검색 로직 (RRF, 재순위화, LLM 호출)
│   └── document_service.py     # 문서 로딩, 전처리, 청킹, 임베딩 저장
├── data_storage/               # 업로드된 원본 파일 임시 저장
├── chroma_db/                  # Chroma 벡터 DB 영속 저장소
├── main.py                     # FastAPI 앱 진입점 및 lifespan 초기화
├── requirements.txt            # 의존성 목록
└── capstone_festival.ipynb     # Google Colab 실행용 노트북
```

---

## 8. API 엔드포인트

### `POST /chatbot`

저장된 문서를 기반으로 질문에 답변합니다.

**Request**
```json
{
  "question": "인수인계 관련 질문",
  "space_id": "my-team"
}
```

**Response**
```json
{
  "answer": "문서 기반 답변 내용",
  "sources": [
    { "source": "인계문서.pdf", "page": 3 }
  ]
}
```

---

### `POST /ingest`

문서를 업로드하고 벡터 DB에 임베딩합니다. 임베딩은 백그라운드에서 비동기 처리되므로, 요청은 즉시 반환됩니다.

**Request (Form-Data)**

| 필드 | 타입 | 설명 |
|------|------|------|
| `file` | File | 업로드할 문서 (PDF, TXT, MD, DOCX) |
| `space_id` | string | 문서를 귀속시킬 스페이스 ID |
| `user_id` | string | 요청자 ID (선택) |

**Response**
```json
{
  "status": "processing",
  "message": "파일 접수 완료. 백그라운드에서 학습 중입니다."
}
```

---

## 9. 실행 방법

현재 Google Colab 환경을 기준으로 구성되어 있습니다. `capstone_festival.ipynb`를 순서대로 실행하세요.

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Ollama 설치 및 모델 다운로드 (로컬 실행 시)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2:9b
ollama pull mxbai-embed-large

# 3. 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```