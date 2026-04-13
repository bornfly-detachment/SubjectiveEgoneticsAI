"""
PRVSE V 层 API — 完整 CRUD

GET    /prvse/v/functions              列出所有函数（系统+用户）
POST   /prvse/v/functions              新建用户自定义函数
PATCH  /prvse/v/functions/{id}         编辑任意字段
DELETE /prvse/v/functions/{id}         删除（仅用户函数）
POST   /prvse/v/compute                手动触发 reward 计算
GET    /prvse/v/history/{task_id}      任务 reward 历史
GET    /prvse/v/stats                  近期统计
"""
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from modules.prvse.v import VRegistry, ExecutionContext, compute_reward
from modules.prvse.v.registry import VFunction
from store.db import get_conn, gen_id

router = APIRouter()
logger = logging.getLogger(__name__)


# ── GET /prvse/v/functions ────────────────────────────────────────────────────

@router.get("/functions")
async def list_functions():
    return VRegistry.list_functions()


# ── POST /prvse/v/functions  新建用户函数 ─────────────────────────────────────

class CreateRequest(BaseModel):
    name:        str
    description: str  = ""
    unit:        str  = "score"
    weight:      float = 1.0
    trigger:     str  = "any"
    expression:  str  = ""    # Python 表达式，ctx = ExecutionContext


@router.post("/functions", status_code=201)
async def create_function(req: CreateRequest):
    if not req.name.strip():
        raise HTTPException(400, "name is required")
    if not req.expression.strip():
        raise HTTPException(400, "expression is required for user functions")

    # 语法检查
    try:
        compile(req.expression, "<expr>", "eval")
    except SyntaxError as e:
        raise HTTPException(400, f"expression syntax error: {e}")

    fid = req.name.strip().replace(" ", "_").lower()
    if fid in {vf["id"] for vf in VRegistry.list_functions()}:
        raise HTTPException(409, f"function '{fid}' already exists")

    now = datetime.now().isoformat()
    conn = get_conn()
    conn.execute(
        """INSERT INTO v_functions
           (id, name, description, unit, weight, trigger, expression, enabled, is_builtin, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,1,0,?,?)""",
        [fid, req.name, req.description, req.unit, req.weight,
         req.trigger, req.expression, now, now]
    )
    conn.commit()
    conn.close()

    # 热加载进内存
    VRegistry.add_user_function(VFunction(
        id=fid, name=req.name, description=req.description,
        unit=req.unit, weight=req.weight, trigger=req.trigger,
        enabled=True, is_builtin=False, expression=req.expression,
    ))

    logger.info(f"[V] created user function: {fid}")
    return {"id": fid, "name": req.name, "is_builtin": False}


# ── PATCH /prvse/v/functions/{id}  编辑 ──────────────────────────────────────

class UpdateRequest(BaseModel):
    description: Optional[str]   = None
    unit:        Optional[str]   = None
    weight:      Optional[float] = None
    trigger:     Optional[str]   = None
    expression:  Optional[str]   = None
    enabled:     Optional[bool]  = None


@router.patch("/functions/{function_id}")
async def update_function(function_id: str, req: UpdateRequest):
    conn = get_conn()
    row = conn.execute("SELECT * FROM v_functions WHERE id=?", [function_id]).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, f"function '{function_id}' not found")

    row = dict(row)
    updates = []
    params  = []

    if req.description is not None:
        updates.append("description=?"); params.append(req.description)
    if req.unit is not None:
        updates.append("unit=?"); params.append(req.unit)
    if req.weight is not None:
        if req.weight < 0:
            conn.close(); raise HTTPException(400, "weight must be >= 0")
        updates.append("weight=?"); params.append(req.weight)
    if req.trigger is not None:
        updates.append("trigger=?"); params.append(req.trigger)
    if req.expression is not None:
        if row["is_builtin"]:
            conn.close(); raise HTTPException(400, "cannot edit expression of builtin function")
        try:
            compile(req.expression, "<expr>", "eval")
        except SyntaxError as e:
            conn.close(); raise HTTPException(400, f"expression syntax error: {e}")
        updates.append("expression=?"); params.append(req.expression)
    if req.enabled is not None:
        updates.append("enabled=?"); params.append(1 if req.enabled else 0)

    if not updates:
        conn.close()
        return {"ok": True, "changed": False}

    updates.append("updated_at=?"); params.append(datetime.now().isoformat())
    params.append(function_id)
    conn.execute(f"UPDATE v_functions SET {', '.join(updates)} WHERE id=?", params)
    conn.commit()
    conn.close()

    # 热更新内存
    VRegistry.reload_all_from_db()
    logger.info(f"[V] updated function: {function_id}")
    return {"ok": True, "id": function_id}


# ── DELETE /prvse/v/functions/{id}  删除（仅用户函数）────────────────────────

@router.delete("/functions/{function_id}")
async def delete_function(function_id: str):
    conn = get_conn()
    row = conn.execute("SELECT is_builtin FROM v_functions WHERE id=?", [function_id]).fetchone()
    if not row:
        conn.close(); raise HTTPException(404, f"function '{function_id}' not found")
    if dict(row)["is_builtin"]:
        conn.close(); raise HTTPException(400, "cannot delete builtin function — use enabled=false to disable")

    conn.execute("DELETE FROM v_functions WHERE id=?", [function_id])
    conn.commit()
    conn.close()

    VRegistry.remove_function(function_id)
    logger.info(f"[V] deleted user function: {function_id}")
    return {"ok": True, "id": function_id}


# ── POST /prvse/v/compute ─────────────────────────────────────────────────────

class ComputeRequest(BaseModel):
    task_id:       str   = ""
    trajectory_id: str   = ""
    node_id:       str   = ""
    node_kind:     str   = "entity"
    status:        str   = "success"
    cost_vector:   dict  = {}
    output_result: dict  = {}
    net_time_ms:   int   = 0
    budget_tokens: int   = 4000
    error_info:    str   = ""


@router.post("/compute")
async def compute(req: ComputeRequest):
    ctx = ExecutionContext(
        task_id=req.task_id, trajectory_id=req.trajectory_id,
        node_id=req.node_id, node_kind=req.node_kind, status=req.status,
        cost_vector=req.cost_vector, output_result=req.output_result,
        net_time_ms=req.net_time_ms, budget_tokens=req.budget_tokens,
        error_info=req.error_info,
    )
    result = compute_reward(ctx)
    return {"total_reward": result.total_reward, "functions": result.functions}


# ── GET /prvse/v/history/{task_id} ────────────────────────────────────────────

@router.get("/history/{task_id}")
async def reward_history(task_id: str, limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        """SELECT id, node_id, node_kind, status, reward, net_time_ms, cost_vector, created_at
           FROM trajectories
           WHERE task_id=? AND reward IS NOT NULL
           ORDER BY created_at DESC LIMIT ?""",
        [task_id, limit]
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── GET /prvse/v/stats ────────────────────────────────────────────────────────

@router.get("/stats")
async def reward_stats(limit: int = 200):
    conn = get_conn()
    row = conn.execute(
        """SELECT
               COUNT(*) as total,
               AVG(reward) as avg_reward,
               MIN(reward) as min_reward,
               MAX(reward) as max_reward,
               SUM(CASE WHEN reward >= 2.0 THEN 1 ELSE 0 END) as high_count,
               SUM(CASE WHEN reward < 0.5  THEN 1 ELSE 0 END) as low_count
           FROM (SELECT reward FROM trajectories WHERE reward IS NOT NULL ORDER BY created_at DESC LIMIT ?)""",
        [limit]
    ).fetchone()
    conn.close()
    return dict(row) if row else {}
