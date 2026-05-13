from typing import List, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """채팅 요청을 위한 데이터 전송 객체(DTO)입니다."""
    question: str
    space_id: Optional[str] = None

class SourceInfo(BaseModel):
    """답변 생성에 사용된 문서 출처 정보를 나타냅니다."""
    source: str
    page: Optional[int] = None

class ChatResponse(BaseModel):
    """채팅 응답을 위한 데이터 전송 객체(DTO)입니다."""
    answer: str
    sources: List[SourceInfo]

class IngestRequest(BaseModel):
    """문서 학습(임베딩) 요청을 위한 데이터 전송 객체(DTO)입니다."""
    space_id: str
    file_path: str