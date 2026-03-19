"""
Lifecycle management: start / stop / status / feedback resolution.
POST /lifecycle/start  — compile task + launch agent loop (LangGraph)
POST /lifecycle/stop   — stop running task
GET  /lifecycle/status/{task_id}
POST /lifecycle/feedback/{feedback_id} — resolve human_gate (LangGraph resume)
WS   /ws/{task_id}     — real-time execution events
"""
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from agent.translator import translate
from agent.loop import AgentLoop
from modules.action import ActionModule
from modules.judge import JudgeModule
from store.db import get_conn, row_to_dict, resolve_feedback
from api.ws import ws_manager
from client.egonetics import egonetics

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Shared module instances ───────────────────────────────────────────────────

_action = ActionModule()
_judge  = JudgeModule()

# Running tasks: task_id → {canvas_id, loop: AgentLoop, task: asyncio.Task}
_running: dict[str, dict] = {}


def _make_loop(task_id: str) -> AgentLoop:
    """Create an AgentLoop wired to WebSocket emit."""
    async def emit(event: str, data: dict):
        await ws_manager.emit(task_id, event, data)

    return AgentLoop(_action, _judge, emit_fn=emit)


# ── Models ────────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    task_id: str
    canvas_id: Optional[str] = None   # None → auto-compile via NL2ExecGraph
    resources: dict = {}


class FeedbackRequest(BaseModel):
    user_response: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/start")
async def lifecycle_start(req: StartRequest):
    if req.task_id in _running:
        raise HTTPException(400, "Task already running")

    canvas_id = req.canvas_id

    if not canvas_id:
        await ws_manager.emit(req.task_id, "compiling", {"message": "NL2ExecGraph Compiler 正在编译任务…"})
        try:
            canvas_id = await translate(req.task_id, _action)
        except RuntimeError as e:
            # Meta-control block
            raise HTTPException(403, str(e))
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            raise HTTPException(500, f"任务编译失败: {e}")

    await ws_manager.emit(req.task_id, "graph_ready", {
        "canvas_id": canvas_id,
        "message": "执行图就绪，LangGraph 开始执行",
    })

    loop = _make_loop(req.task_id)
    loop_task = asyncio.create_task(_run(loop, req.task_id, canvas_id))
    _running[req.task_id] = {"canvas_id": canvas_id, "loop": loop, "task": loop_task}

    return {"status": "started", "task_id": req.task_id, "canvas_id": canvas_id}


async def _run(loop: AgentLoop, task_id: str, canvas_id: str):
    try:
        await loop.run(task_id, canvas_id)
    except asyncio.CancelledError:
        logger.info(f"Loop cancelled: task={task_id}")
    except Exception as e:
        logger.error(f"Loop error task={task_id}: {e}")
        await ws_manager.emit(task_id, "task_failed", {"error": str(e)})
    finally:
        _running.pop(task_id, None)


@router.post("/stop/{task_id}")
async def lifecycle_stop(task_id: str):
    entry = _running.pop(task_id, None)
    if not entry:
        raise HTTPException(404, "Task not running")
    entry["task"].cancel()
    await egonetics.update_task(task_id, {"column_id": "planned", "task_summary": "已手动终止"})
    await ws_manager.emit(task_id, "task_stopped", {"message": "任务已终止"})
    return {"status": "stopped", "task_id": task_id}


@router.get("/status/{task_id}")
async def lifecycle_status(task_id: str):
    conn = get_conn()
    trajs = conn.execute(
        "SELECT * FROM trajectories WHERE task_id=? ORDER BY created_at DESC LIMIT 30",
        [task_id]
    ).fetchall()
    feedback = conn.execute(
        "SELECT * FROM user_feedback WHERE task_id=? AND resolved=0 ORDER BY created_at",
        [task_id]
    ).fetchall()
    conn.close()
    return {
        "task_id":          task_id,
        "running":          task_id in _running,
        "canvas_id":        _running.get(task_id, {}).get("canvas_id"),
        "trajectories":     [row_to_dict(r) for r in trajs],
        "pending_feedback": [row_to_dict(r) for r in feedback],
    }


@router.post("/feedback/{feedback_id}")
async def resolve_human_gate(feedback_id: str, req: FeedbackRequest):
    """
    Resolve a human_gate or decision_query feedback.
    If the issuing task is still in _running, resume the LangGraph via Command(resume=...).
    """
    resolve_feedback(feedback_id, req.user_response)

    conn = get_conn()
    row = conn.execute("SELECT task_id FROM user_feedback WHERE id=?", [feedback_id]).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Feedback not found")

    task_id = dict(row)["task_id"]
    await ws_manager.emit(task_id, "feedback_resolved", {
        "feedback_id": feedback_id,
        "response": req.user_response,
    })

    # Resume LangGraph if task is still managed
    entry = _running.get(task_id)
    if entry:
        loop: AgentLoop = entry["loop"]
        canvas_id = entry["canvas_id"]
        # Run resume as a new task (the old task is blocked on interrupt)
        asyncio.create_task(_resume(loop, task_id, canvas_id, req.user_response))

    return {"status": "resolved", "feedback_id": feedback_id, "task_id": task_id}


async def _resume(loop: AgentLoop, task_id: str, canvas_id: str, feedback_value: str):
    try:
        await loop.resume(task_id, canvas_id, feedback_value)
    except Exception as e:
        logger.error(f"Resume error task={task_id}: {e}")
        await ws_manager.emit(task_id, "task_failed", {"error": str(e)})


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await ws_manager.connect(task_id, websocket)
    entry = _running.get(task_id)
    await websocket.send_json({
        "type": "connected",
        "task_id": task_id,
        "running": bool(entry),
        "canvas_id": entry["canvas_id"] if entry else None,
    })
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(task_id, websocket)
