"""Agent control: start/stop/status for task execution."""
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from modules.action import ActionModule
from modules.env_feedback import EnvFeedbackModule
from modules.judge import JudgeModule
from agent.executor import NodeExecutor
from agent.loop import AgentLoop

router = APIRouter()

# Shared instances
_action = ActionModule()
_judge = JudgeModule()
_feedback_env = EnvFeedbackModule(_action)

async def _notify_user(feedback_id: str, task_id: str, message: str):
    # TODO: integrate with Egonetics notification system
    import logging
    logging.getLogger(__name__).info(f"[HUMAN_GATE] task={task_id} fb={feedback_id}: {message}")

_executor = NodeExecutor(_action, _judge, _notify_user)
_loop = AgentLoop(_executor)
_running_tasks: dict = {}


class RunTaskRequest(BaseModel):
    task_id: str
    canvas_id: str


@router.post("/run")
async def run_task(req: RunTaskRequest, bg: BackgroundTasks):
    if req.task_id in _running_tasks:
        raise HTTPException(400, "Task already running")
    _running_tasks[req.task_id] = "running"
    bg.add_task(_run_and_cleanup, req.task_id, req.canvas_id)
    return {"status": "started", "task_id": req.task_id}


async def _run_and_cleanup(task_id: str, canvas_id: str):
    try:
        await _loop.run_task(task_id, canvas_id)
    finally:
        _running_tasks.pop(task_id, None)


@router.get("/status/{task_id}")
async def task_status(task_id: str):
    from store.db import get_conn, row_to_dict
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM trajectories WHERE task_id=? ORDER BY created_at DESC LIMIT 20",
        [task_id]
    ).fetchall()
    conn.close()
    return {
        "task_id": task_id,
        "running": task_id in _running_tasks,
        "trajectories": [row_to_dict(r) for r in rows]
    }


@router.post("/stop/{task_id}")
async def stop_task(task_id: str):
    _running_tasks.pop(task_id, None)
    _loop.backtrack.reset(task_id)
    return {"status": "stopped", "task_id": task_id}
