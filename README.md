# Capstone Chatbot API

RAG(Retrieval-Augmented Generation) 기반의 문서 챗봇입니다.
구글 Colab 환경 또는 로컬에서 구동할 수 있도록 설계되었습니다. (현재는 구글 Colab에 맞춰져 있으며, 코랩으로 맞춘 후 로컬 환경은 확인 예정)

## 주요 기능

- **문서 임베딩 (`/ingest`)**: PDF, TXT, DOCX, PPTX, XLSX 등 다양한 포맷의 문서를 업로드하여 벡터 데이터베이스(Chroma)에 저장합니다. 백그라운드에서 비동기적으로 처리됩니다.
- **RAG 기반 질의응답 (`/chatbot`)**: 사용자의 질문에 대해 저장된 문서를 검색하여 기반 지식으로 활용해 답변을 생성하며, 출처와 페이지 정보를 함께 제공합니다.
- **하이브리드 검색**: Dense Retrieval(임베딩 기반 코사인 유사도 검색)과 Sparse Retrieval(BM25 기반 키워드 검색)을 결합하여 검색 품질을 높였습니다.

## 기술 스택

- **웹 프레임워크**: FastAPI
- **LLM / 파이프라인**: LangChain, Ollama (모델: `gemma2:9b`)
- **벡터 데이터베이스**: Chroma
- **임베딩 모델**: `mxbai-embed-large`

## 설치 및 실행 방법

1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama 설치 및 모델 다운로드** (로컬 실행 시 필요)
   ```bash
   ollama pull gemma2:9b
   ollama pull mxbai-embed-large
   ```

3. **서버 실행**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 주요 API 엔드포인트

### `POST /chatbot`
- 사용자의 질문을 받아 RAG 기반 답변을 반환합니다.
- **Request Body (JSON)**: 
  ```json
  {
    "question": "질문할 내용",
    "space_id": "default"
  }
  ```

### `POST /ingest`
- 문서를 서버에 업로드하고 벡터 DB에 임베딩합니다.
- **Request Body (Form-Data)**: 
  - `file`: 업로드할 파일 객체
  - `space_id`: 문서를 구분할 공간/그룹 ID
  - `user_id`: 요청자 ID (선택 사항)

## 프로젝트 구조

```text
.
├── api/
│   └── routes.py           # API 라우팅 (컨트롤러) 정의
├── core/
│   └── config.py           # 환경 변수, 프롬프트, 모델 등 전역 설정
├── schemas/
│   └── dto.py              # 데이터 전송 객체 (Pydantic 모델)
├── services/
│   └── chatbot_service.py  # 핵심 비즈니스 로직 (검색, 임베딩, LLM 호출)
├── main.py                 # FastAPI 애플리케이션 진입점
└── requirements.txt        # 패키지 의존성 목록
```
