import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.config import DB_DIR, ensure_dir, get_vectorstore
from api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 생명주기를 관리합니다.
    서버 시작 시 필요한 디렉토리를 생성하고 벡터 데이터베이스를 초기화합니다.
    """
    ensure_dir(DB_DIR)

    app.state.vectordb = get_vectorstore()
    app.state.bm25 = None
    app.state.write_lock = asyncio.Lock()
    app.state.rebuild_lock = asyncio.Lock()
    app.state.stop_event = asyncio.Event()

    print(">> 서버가 시작되었습니다.")
    yield

    app.state.stop_event.set()
    print(">> 서버가 종료되었습니다.")

app = FastAPI(title="RAG API (Layered)", lifespan=lifespan)

# 라우터 등록
app.include_router(router)