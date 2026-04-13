"""
Main API service for SubjectiveEgoneticsAI.
FastAPI on port 8000. Egonetics frontend/backend communicates with this.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from store.db import init_db
from api.routes import agent, feedback, llm as llm_routes
from api.routes import model as model_routes
from api.routes import lifecycle as lifecycle_routes
from api.routes import prvse_v
from api.routes import e0 as e0_routes
from api.routes import state_ctrl as state_ctrl_routes
from api.routes import queue as queue_routes
from api.routes import plan as plan_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # V 层初始化：注册系统函数 → upsert DB → 加载用户函数
    from modules.prvse.v import VRegistry
    VRegistry.sync_builtins_to_db()
    VRegistry.load_user_functions_from_db()
    # S 层初始化：同步内置生命周期 + PRVSE 组件到 DB
    from modules.prvse.s import E0LifecycleManager
    E0LifecycleManager.sync_builtins_to_db()
    # State Controller：24/7 任务队列轮询
    from agent.state_controller import state_controller
    import asyncio
    ctrl_task = asyncio.create_task(state_controller.start(), name="state-controller")
    logger.info("SubjectiveEgoneticsAI API started")
    yield
    await state_controller.stop()
    ctrl_task.cancel()
    logger.info("SubjectiveEgoneticsAI API stopped")


app = FastAPI(
    title="SubjectiveEgoneticsAI",
    description="Self-cybernetics agent execution engine",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router,            prefix="/agent",     tags=["agent"])
app.include_router(feedback.router,         prefix="/feedback",  tags=["feedback"])
app.include_router(model_routes.router,     prefix="/model",     tags=["model"])
app.include_router(lifecycle_routes.router, prefix="/lifecycle", tags=["lifecycle"])
app.include_router(llm_routes.router,       prefix="/llm",       tags=["llm"])
app.include_router(prvse_v.router,          prefix="/prvse/v",   tags=["prvse-v"])
app.include_router(e0_routes.router,         prefix="/e0",          tags=["e0"])
app.include_router(state_ctrl_routes.router, prefix="/state-ctrl",  tags=["state-ctrl"])
app.include_router(queue_routes.router,      prefix="/queue",        tags=["queue"])
app.include_router(plan_routes.router,       prefix="/plan",         tags=["plan"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "SubjectiveEgoneticsAI"}
