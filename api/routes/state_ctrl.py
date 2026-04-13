"""
State Controller status + control routes.
GET  /state-ctrl/status   控制器运行状态
GET  /state-ctrl/events   轨迹事件流（轮询）
POST /state-ctrl/pause    暂停执行
POST /state-ctrl/resume   恢复执行
GET  /state-ctrl/stream   SSE 实时状态推送
"""
import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from agent.state_controller import state_controller
from store.db import get_conn, row_to_dict

router = APIRouter()


# ─── Status ───────────────────────────────────────────────────────────────────

@router.get("/status")
async def status():
    s = state_controller.status()
    conn = get_conn()
    try:
        pending = conn.execute(
            "SELECT COUNT(*) as n FROM execution_queue WHERE state='pending' AND (layer='local' OR layer IS NULL)"
        ).fetchone()["n"]
        running = conn.execute(
            "SELECT COUNT(*) as n FROM execution_queue WHERE state='running' AND (layer='local' OR layer IS NULL)"
        ).fetchone()["n"]
    except Exception:
        pending = conn.execute(
            "SELECT COUNT(*) as n FROM execution_queue WHERE state='pending'"
        ).fetchone()["n"]
        running = conn.execute(
            "SELECT COUNT(*) as n FROM execution_queue WHERE state='running'"
        ).fetchone()["n"]
    conn.close()
    return {**s, "pending_count": pending, "running_count": running}


# ─── Control ──────────────────────────────────────────────────────────────────

@router.post("/pause")
async def pause():
    state_controller.pause()
    return {"ok": True, "is_paused": True}


@router.post("/resume")
async def resume():
    state_controller.resume()
    return {"ok": True, "is_paused": False}


# ─── Events（轮询模式）────────────────────────────────────────────────────────

@router.get("/events")
async def events(
    task_ref_id: Optional[str] = Query(None),
    limit: int = Query(30, le=100),
):
    conn = get_conn()
    if task_ref_id:
        rows = conn.execute(
            """SELECT id, task_id, canvas_id, node_id, node_kind,
                      status, started_at, ended_at, cost_vector, reward, error_info
               FROM trajectories WHERE task_id=?
               ORDER BY created_at DESC LIMIT ?""",
            [task_ref_id, limit]
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, task_id, canvas_id, node_id, node_kind,
                      status, started_at, ended_at, cost_vector, reward, error_info
               FROM trajectories ORDER BY created_at DESC LIMIT ?""",
            [limit]
        ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


# ─── SSE 实时状态流 ───────────────────────────────────────────────────────────

@router.get("/stream")
async def stream():
    """SSE：每 3 秒推送一次控制器状态快照"""

    async def generate():
        while True:
            try:
                s = state_controller.status()
                conn = get_conn()
                try:
                    pending = conn.execute(
                        "SELECT COUNT(*) as n FROM execution_queue WHERE state='pending' AND (layer='local' OR layer IS NULL)"
                    ).fetchone()["n"]
                    running = conn.execute(
                        "SELECT COUNT(*) as n FROM execution_queue WHERE state='running' AND (layer='local' OR layer IS NULL)"
                    ).fetchone()["n"]
                    conflicts = conn.execute(
                        "SELECT COUNT(*) as n FROM execution_queue WHERE layer='conflict' AND resolved=0"
                    ).fetchone()["n"]
                except Exception:
                    pending = running = conflicts = 0
                conn.close()

                payload = json.dumps({
                    **s,
                    "pending_count":   pending,
                    "running_count":   running,
                    "conflict_count":  conflicts,
                }, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            await asyncio.sleep(3)

    return StreamingResponse(generate(), media_type="text/event-stream")
