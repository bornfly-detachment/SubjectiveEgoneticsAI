"""
Main API service for SubjectiveEgoneticsAI.
FastAPI on port 8000. Egonetics frontend/backend communicates with this.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from store.db import init_db
from api.routes import agent, feedback
from api.routes import model as model_routes
from api.routes import lifecycle as lifecycle_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("SubjectiveEgoneticsAI API started")
    yield
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

app.include_router(agent.router,           prefix="/agent",     tags=["agent"])
app.include_router(feedback.router,        prefix="/feedback",  tags=["feedback"])
app.include_router(model_routes.router,    prefix="/model",     tags=["model"])
app.include_router(lifecycle_routes.router, prefix="/lifecycle", tags=["lifecycle"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "SubjectiveEgoneticsAI"}
