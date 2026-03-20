"""Agent session status — read-only view of trajectories."""
from fastapi import APIRouter
from store.db import get_conn, row_to_dict

router = APIRouter()


@router.get("/status/{task_id}")
async def task_status(task_id: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM trajectories WHERE task_id=? ORDER BY created_at DESC LIMIT 30",
        [task_id]
    ).fetchall()
    conn.close()
    return {"task_id": task_id, "trajectories": [row_to_dict(r) for r in rows]}
