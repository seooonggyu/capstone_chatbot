import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.config import DB_DIR, ensure_dir, get_vectorstore
from api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dir(DB_DIR)

    app.state.vectordb = get_vectorstore()
    app.state.bm25 = {}
    app.state.write_lock = asyncio.Lock()
    app.state.rebuild_lock = asyncio.Lock()
    app.state.stop_event = asyncio.Event()

    print(">> Server Started (Layered Architecture)")
    yield

    app.state.stop_event.set()
    print(">> Server Shutdown")

app = FastAPI(title="RAG API (Layered)", lifespan=lifespan)

app.include_router(router)