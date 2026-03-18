"""User feedback endpoints: 4 types."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from store.db import create_feedback, resolve_feedback, get_pending_feedback, get_conn, row_to_dict

router = APIRouter()


class FeedbackResolveRequest(BaseModel):
    user_response: str


class FeedbackCreateRequest(BaseModel):
    task_id: str
    feedback_type: str  # graph_update | failure_analysis | decision_query | value_judgment
    context: dict = {}
    prompt: str = ""
    is_blocking: bool = True


@router.post("/")
async def create(req: FeedbackCreateRequest):
    fb_id = create_feedback(req.task_id, req.feedback_type, req.context, req.prompt, req.is_blocking)
    return {"id": fb_id}


@router.get("/pending/{task_id}")
async def pending(task_id: str):
    return get_pending_feedback(task_id)


@router.get("/all")
async def list_all(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM user_feedback ORDER BY created_at DESC LIMIT ?", [limit]
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@router.patch("/{feedback_id}/resolve")
async def resolve(feedback_id: str, req: FeedbackResolveRequest):
    resolve_feedback(feedback_id, req.user_response)
    return {"status": "resolved", "id": feedback_id}


@router.get("/failure-cases")
async def failure_cases(analyzed: bool = False):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM failure_cases WHERE analyzed=? ORDER BY created_at DESC LIMIT 50",
        [1 if analyzed else 0]
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@router.patch("/failure-cases/{case_id}")
async def update_failure_case(case_id: str, body: dict):
    conn = get_conn()
    updates = []
    params = []
    if "root_cause" in body:
        updates.append("root_cause=?"); params.append(body["root_cause"])
    if "solution" in body:
        updates.append("solution=?"); params.append(body["solution"])
    updates.append("analyzed=1")
    params.append(case_id)
    conn.execute(f"UPDATE failure_cases SET {', '.join(updates)} WHERE id=?", params)
    conn.commit(); conn.close()
    return {"status": "updated"}
