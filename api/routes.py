from fastapi import APIRouter, Request
from fastapi import UploadFile, File, Form, BackgroundTasks
from schemas.dto import ChatRequest, ChatResponse, IngestRequest
from services.chatbot_service import process_chat, process_ingest

router = APIRouter()

@router.post("/chatbot", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """
    사용자의 질문을 받아 RAG 모델을 통해 답변을 생성하고 반환합니다.
    """
    return await process_chat(req, request.app)

@router.post("/ingest")
async def ingest_file(
    request: Request, 
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    space_id: str = Form(...), 
    user_id: str = Form("Unknown")
):
    """
    사용자가 업로드한 파일을 저장하고, 백그라운드에서 벡터 데이터베이스에 임베딩합니다.
    """
    save_path = f"/content/data_storage/{file.filename}"
    with open(save_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    background_tasks.add_task(process_ingest, save_path, space_id, request.app, user_id)

    return {"status": "processing", "message": "파일 접수가 완료되었습니다. 백그라운드에서 문서 처리가 진행됩니다."}