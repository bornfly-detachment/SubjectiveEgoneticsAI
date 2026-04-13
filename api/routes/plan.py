"""
Controller Plan routes — 三层控制器：全局计划 + 冲突处理
POST /plan/generate             LLM 生成全局计划 DAG
GET  /plan/                     列出全局计划节点
POST /plan/confirm              确认计划 → 自动流入局部层
PATCH /plan/{id}                手动编辑计划节点
DELETE /plan/{id}               删除计划节点
GET  /plan/conflicts            列出未解决冲突
POST /plan/conflicts/{id}/resolve  解决冲突
"""
import json
import logging
from datetime import datetime
from typing import Optional

import anthropic
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import settings
from store.db import get_conn, gen_id, row_to_dict

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── LLM helper ──────────────────────────────────────────────────────────────

def _make_client() -> anthropic.AsyncAnthropic:
    kwargs: dict = {"api_key": settings.anthropic_api_key}
    if settings.anthropic_base_url:
        kwargs["base_url"] = settings.anthropic_base_url
    if settings.llm_proxy:
        kwargs["http_client"] = httpx.AsyncClient(proxy=settings.llm_proxy)
    return anthropic.AsyncAnthropic(**kwargs)


PLAN_SYSTEM = """你是一个任务规划专家。用户给你一段任务描述，你需要将其拆解为一个有向无环图（DAG）执行计划。

输出格式（只输出 JSON，不要解释）：
{
  "nodes": [
    {"id": "p1", "label": "步骤名称（10字以内）", "dependencies": []},
    {"id": "p2", "label": "步骤名称", "dependencies": ["p1"]},
    ...
  ]
}

规则：
- 节点数量：3~8 个
- label 必须简洁（中文，10字以内）
- dependencies 是该节点依赖的节点 id 列表（空列表 = 根节点）
- 不得有循环依赖
- 只输出 JSON，不要 markdown 代码块"""


# ─── Schema ──────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    description: str
    task_ref_id: Optional[str] = None  # 关联的 Egonetics task ID


class PatchNodeRequest(BaseModel):
    label: Optional[str] = None
    dependencies: Optional[list[str]] = None

class RefineNodeRequest(BaseModel):
    feedback: str  # 用户的修改意见


class ResolveRequest(BaseModel):
    option_index: int  # 用户选择的选项下标


# ─── 生成计划 ─────────────────────────────────────────────────────────────────

@router.post("/generate")
async def generate_plan(body: GenerateRequest):
    """LLM 生成全局计划 DAG，写入 execution_queue（layer='global'）"""

    # 1. 调用 LLM
    try:
        client = _make_client()
        msg = await client.messages.create(
            model=settings.default_llm_model,
            max_tokens=1024,
            system=PLAN_SYSTEM,
            messages=[{"role": "user", "content": body.description}],
        )
        raw = msg.content[0].text.strip()
        # 去掉可能的 markdown 包裹
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        plan = json.loads(raw)
    except Exception as e:
        logger.error("Plan generation failed: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    nodes = plan.get("nodes", [])
    if not nodes:
        raise HTTPException(status_code=422, detail="LLM returned empty plan")

    # 2. 先清理同一 task_ref_id 的旧全局计划（如果是重新生成）
    conn = get_conn()
    if body.task_ref_id:
        old_nodes = conn.execute(
            "SELECT id, regen_count FROM execution_queue WHERE task_ref_id=? AND layer='global'",
            [body.task_ref_id]
        ).fetchall()
        if old_nodes:
            regen_count = old_nodes[0]["regen_count"] + 1
            # 检查是否超限
            max_regen = old_nodes[0]["max_regen"] if "max_regen" in old_nodes[0].keys() else 3
            if regen_count > max_regen:
                conn.close()
                raise HTTPException(status_code=429, detail="Max regeneration count reached")
            conn.execute(
                "DELETE FROM execution_queue WHERE task_ref_id=? AND layer='global'",
                [body.task_ref_id]
            )
            conn.commit()
        else:
            regen_count = 0
    else:
        regen_count = 0

    # 3. 写入新节点
    inserted = []
    for node in nodes:
        node_id = gen_id("gp")
        conn.execute(
            """INSERT INTO execution_queue
               (id, description, layer, dependencies, regen_count, max_regen, task_ref_id, sort_order, state)
               VALUES (?, ?, 'global', ?, ?, 3, ?, ?, 'pending')""",
            [
                node_id,
                node["label"],
                json.dumps(node.get("dependencies", [])),
                regen_count,
                body.task_ref_id,
                nodes.index(node),
            ]
        )
        inserted.append({**node, "db_id": node_id, "regen_count": regen_count})

    conn.commit()
    conn.close()

    return {"nodes": inserted, "regen_count": regen_count, "max_regen": 3}


# ─── 列出全局计划 ─────────────────────────────────────────────────────────────

@router.get("/")
async def list_plan(task_ref_id: Optional[str] = None):
    conn = get_conn()
    if task_ref_id:
        rows = conn.execute(
            "SELECT * FROM execution_queue WHERE layer='global' AND task_ref_id=? ORDER BY sort_order",
            [task_ref_id]
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM execution_queue WHERE layer='global' ORDER BY sort_order"
        ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


# ─── 确认计划 → 创建 Egonetics Canvas 节点 + 流入局部层 ─────────────────────

@router.post("/confirm")
async def confirm_plan(task_ref_id: Optional[str] = None):
    """
    1. 在 Egonetics 创建执行画布，每个计划节点 → llm_call 画布节点
    2. 按依赖关系创建画布节点间的 relations
    3. 创建局部层 execution_queue 条目，绑定 canvas_id
    """
    from client.egonetics import egonetics

    conn = get_conn()
    q = "SELECT * FROM execution_queue WHERE layer='global' AND state='pending'"
    params: list = []
    if task_ref_id:
        q += " AND task_ref_id=?"
        params.append(task_ref_id)

    nodes = [row_to_dict(r) for r in conn.execute(q, params).fetchall()]
    if not nodes:
        conn.close()
        return {"created": 0, "local_ids": [], "canvas_id": None}

    # 1. 创建执行画布
    title = f"执行: {nodes[0].get('description', task_ref_id or 'plan')[:30]}"
    try:
        canvas = await egonetics.create_execution_canvas(task_ref_id or gen_id("cvs"), title)
        canvas_id = canvas["id"]
    except Exception as e:
        logger.warning("create_execution_canvas failed: %s — fallback to local id", e)
        canvas_id = gen_id("cvs")

    # 2. 为每个全局计划节点创建 llm_call 画布节点
    # plan_node_db_id → canvas_node_id
    node_canvas_map: dict[str, str] = {}
    COL_W, ROW_H = 220, 100
    for idx, node in enumerate(nodes):
        try:
            cn = await egonetics.add_node(
                canvas_id=canvas_id,
                entity_type="task",
                entity_id=node["id"],
                x=float(idx * COL_W + 80),
                y=80.0,
                node_kind="llm_call",
                exec_config={
                    "prompt": node["description"],
                    "system": "你是一个执行专家，按照任务描述完成具体工作，输出执行结果。",
                    "budget_tokens": 2000,
                },
            )
            node_canvas_map[node["id"]] = cn["id"]
        except Exception as e:
            logger.warning("add_node failed for %s: %s", node["id"], e)

    # 3. 按依赖关系创建 canvas relations（dep → node）
    for node in nodes:
        deps = node.get("dependencies") or []
        if isinstance(deps, str):
            try:
                deps = json.loads(deps)
            except Exception:
                deps = []
        tgt_canvas_id = node_canvas_map.get(node["id"])
        if not tgt_canvas_id:
            continue
        for dep_id in deps:
            src_canvas_id = node_canvas_map.get(dep_id)
            if src_canvas_id:
                try:
                    await egonetics.create_relation(
                        source_type="canvas_node", source_id=src_canvas_id,
                        target_type="canvas_node", target_id=tgt_canvas_id,
                        relation_type="next", title="→",
                    )
                except Exception as e:
                    logger.warning("create_relation failed: %s", e)

    # 4. 创建局部层队列条目，绑定 canvas_id
    created = []
    for node in nodes:
        local_id = gen_id("lq")
        conn.execute(
            """INSERT INTO execution_queue
               (id, description, layer, global_ref_id, task_ref_id, sort_order, state, assigned_node)
               VALUES (?, ?, 'local', ?, ?, ?, 'pending', 'local')""",
            [local_id, node["description"], node["id"],
             node.get("task_ref_id"), node.get("sort_order", 0)]
        )
        # 将 canvas_id 写回全局节点，供后续查询
        conn.execute(
            "UPDATE execution_queue SET output=? WHERE id=?",
            [json.dumps({"canvas_id": canvas_id}), node["id"]]
        )
        created.append(local_id)

    conn.commit()
    conn.close()
    return {"created": len(created), "local_ids": created, "canvas_id": canvas_id}


# ─── 手动编辑节点 ─────────────────────────────────────────────────────────────

@router.patch("/{node_id}")
async def patch_node(node_id: str, body: PatchNodeRequest):
    fields, values = [], []
    if body.label is not None:
        fields.append("description=?"); values.append(body.label)
    if body.dependencies is not None:
        fields.append("dependencies=?"); values.append(json.dumps(body.dependencies))
    if not fields:
        raise HTTPException(400, "No fields to update")
    values.append(node_id)
    conn = get_conn()
    conn.execute(f"UPDATE execution_queue SET {', '.join(fields)} WHERE id=? AND layer='global'", values)
    conn.commit()
    row = conn.execute("SELECT * FROM execution_queue WHERE id=?", [node_id]).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Node not found")
    return row_to_dict(row)


@router.post("/{node_id}/refine")
async def refine_node(node_id: str, body: RefineNodeRequest):
    """根据用户修改意见，用 LLM 优化单个计划节点的描述"""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM execution_queue WHERE id=? AND layer='global'", [node_id]
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Node not found")

    node = row_to_dict(row)
    original = node["description"]

    try:
        client = _make_client()
        msg = await client.messages.create(
            model=settings.default_llm_model,
            max_tokens=256,
            system="你是一个任务规划专家。根据用户的修改意见，优化执行步骤的描述。只输出新的步骤名称，10字以内，不要解释。",
            messages=[{
                "role": "user",
                "content": f"原步骤：{original}\n修改意见：{body.feedback}\n新步骤名称："
            }],
        )
        new_label = msg.content[0].text.strip().replace("\"", "").replace("'", "")[:30]
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")

    conn = get_conn()
    conn.execute(
        "UPDATE execution_queue SET description=? WHERE id=?", [new_label, node_id]
    )
    conn.commit()
    row = conn.execute("SELECT * FROM execution_queue WHERE id=?", [node_id]).fetchone()
    conn.close()
    return {**row_to_dict(row), "original": original}


@router.delete("/{node_id}")
async def delete_node(node_id: str):
    conn = get_conn()
    cur = conn.execute("DELETE FROM execution_queue WHERE id=? AND layer='global'", [node_id])
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise HTTPException(404, "Node not found")
    return {"ok": True}


# ─── 冲突管理 ─────────────────────────────────────────────────────────────────

@router.get("/conflicts")
async def list_conflicts():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM execution_queue WHERE layer='conflict' AND resolved=0 ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(conflict_id: str, body: ResolveRequest):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM execution_queue WHERE id=? AND layer='conflict'", [conflict_id]
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Conflict not found")

    node = row_to_dict(row)
    options = node.get("options") or []
    if body.option_index >= len(options):
        conn.close()
        raise HTTPException(400, "Invalid option index")

    chosen = options[body.option_index]

    conn.execute(
        "UPDATE execution_queue SET resolved=1, completed_at=?, output=? WHERE id=?",
        [datetime.now().isoformat(), json.dumps(chosen), conflict_id]
    )
    conn.commit()
    conn.close()
    return {"ok": True, "chosen": chosen}
