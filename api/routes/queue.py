"""
Execution Queue routes — 机器任务队列 CRUD
POST   /queue/          加入队列
GET    /queue/          列出队列（支持 ?state= 过滤）
GET    /queue/:id       单条
PATCH  /queue/:id       更新状态/输出
DELETE /queue/:id       删除

GET    /queue/preconditions  前提条件健康检查
"""
import httpx
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from store.db import get_conn, gen_id, row_to_dict
from config.settings import settings

router = APIRouter()


# ── Schema ───────────────────────────────────────────────────────────────────

class QueueCreate(BaseModel):
    description: str
    v_criteria: dict = {}
    sort_order: float = 0
    task_ref_id: Optional[str] = None   # Egonetics task ID（桥接用）


class QueuePatch(BaseModel):
    state: Optional[str] = None          # pending|running|done|failed|blocked
    assigned_node: Optional[str] = None
    output: Optional[str] = None
    v_score: Optional[float] = None
    error_msg: Optional[str] = None


# ── CRUD ─────────────────────────────────────────────────────────────────────

@router.post("/")
async def create(body: QueueCreate):
    item_id = gen_id("q")
    conn = get_conn()
    import json
    conn.execute(
        """INSERT INTO execution_queue (id, description, v_criteria, sort_order, task_ref_id)
           VALUES (?, ?, ?, ?, ?)""",
        [item_id, body.description, json.dumps(body.v_criteria), body.sort_order, body.task_ref_id]
    )
    conn.commit()
    row = conn.execute("SELECT * FROM execution_queue WHERE id=?", [item_id]).fetchone()
    conn.close()
    return row_to_dict(row)


@router.get("/")
async def list_queue(state: Optional[str] = None):
    conn = get_conn()
    if state:
        rows = conn.execute(
            "SELECT * FROM execution_queue WHERE state=? ORDER BY sort_order, created_at",
            [state]
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM execution_queue ORDER BY sort_order, created_at"
        ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@router.get("/preconditions")
async def preconditions():
    """四项前提条件健康检查"""
    import shutil, platform, psutil

    # 1. 环境
    env_ok = True
    env_info = {
        "python": platform.python_version(),
        "disk_gb": round(shutil.disk_usage("/").free / 1e9, 1),
        "memory_gb": round(psutil.virtual_memory().available / 1e9, 1),
    }

    # 2. 目标（execution_queue 里有 pending 任务）
    conn = get_conn()
    pending_count = conn.execute(
        "SELECT COUNT(*) as n FROM execution_queue WHERE state='pending'"
    ).fetchone()["n"]
    conn.close()
    goals_ok = pending_count > 0

    # 3. 实践能力（SEAI 自身健康 = 这个接口能响应就表示 ok）
    nodes_ok = True

    # 4. 认知（LLM API 可达）
    cognition_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(settings.inference_url + "/health")
            cognition_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "environment": {"ok": env_ok,      "detail": env_info},
        "goals":       {"ok": goals_ok,     "detail": {"pending": pending_count}},
        "nodes":       {"ok": nodes_ok,     "detail": {"node": "local"}},
        "cognition":   {"ok": cognition_ok, "detail": {"url": settings.inference_url}},
        "all_ok":      env_ok and goals_ok and nodes_ok,
    }


@router.get("/{item_id}")
async def get_item(item_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM execution_queue WHERE id=?", [item_id]).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row_to_dict(row)


@router.patch("/{item_id}")
async def patch_item(item_id: str, body: QueuePatch):
    fields, values = [], []

    if body.state         is not None:
        fields.append("state=?"); values.append(body.state)
        if body.state == "running":
            fields.append("started_at=?"); values.append(datetime.now().isoformat())
        elif body.state in ("done", "failed", "blocked"):
            fields.append("completed_at=?"); values.append(datetime.now().isoformat())
    if body.assigned_node is not None: fields.append("assigned_node=?"); values.append(body.assigned_node)
    if body.output        is not None: fields.append("output=?");        values.append(body.output)
    if body.v_score       is not None: fields.append("v_score=?");       values.append(body.v_score)
    if body.error_msg     is not None: fields.append("error_msg=?");     values.append(body.error_msg)

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    values.append(item_id)
    conn = get_conn()
    conn.execute(f"UPDATE execution_queue SET {', '.join(fields)} WHERE id=?", values)
    conn.commit()
    row = conn.execute("SELECT * FROM execution_queue WHERE id=?", [item_id]).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row_to_dict(row)


@router.delete("/{item_id}")
async def delete_item(item_id: str):
    conn = get_conn()
    cur = conn.execute("DELETE FROM execution_queue WHERE id=?", [item_id])
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}
