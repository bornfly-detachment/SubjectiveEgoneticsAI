"""
E0 Lifecycle + PRVSE Components — 完整 CRUD

GET    /e0/lifecycles                           列出所有生命周期
POST   /e0/lifecycles                           新建自定义生命周期
PATCH  /e0/lifecycles/{id}                      编辑（name/description/enabled）
DELETE /e0/lifecycles/{id}                      删除（仅用户自定义）

GET    /e0/lifecycles/{id}/state                当前状态
POST   /e0/lifecycles/{id}/state/transition     触发状态转换

GET    /e0/components                           列出 PRVSE 组件（?lifecycle_id=&layer=）
POST   /e0/components                           新建组件
PATCH  /e0/components/{id}                      编辑组件
DELETE /e0/components/{id}                      删除（仅用户自定义）
"""
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from modules.prvse.s.lifecycle import VALID_STATES, VALID_TRANSITIONS
from store.db import get_conn

router = APIRouter()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycles CRUD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/lifecycles")
async def list_lifecycles():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM e0_lifecycles ORDER BY is_builtin DESC, id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


class CreateLifecycleRequest(BaseModel):
    id: str
    name: str
    description: str = ""


@router.post("/lifecycles", status_code=201)
async def create_lifecycle(req: CreateLifecycleRequest):
    if not req.id.strip() or not req.name.strip():
        raise HTTPException(400, "id and name are required")
    conn = get_conn()
    try:
        exists = conn.execute("SELECT id FROM e0_lifecycles WHERE id=?", [req.id]).fetchone()
        if exists:
            raise HTTPException(409, f"lifecycle '{req.id}' already exists")
        now = datetime.now().isoformat()
        conn.execute(
            """INSERT INTO e0_lifecycles (id, name, description, state, state_meta, enabled, is_builtin, created_at, updated_at)
               VALUES (?,?,?,'IDLE','{}',1,0,?,?)""",
            [req.id.strip(), req.name.strip(), req.description, now, now],
        )
        conn.commit()
        logger.info(f"[E0] created lifecycle: {req.id}")
        return {"id": req.id, "name": req.name, "state": "IDLE", "is_builtin": False}
    finally:
        conn.close()


class UpdateLifecycleRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None


@router.patch("/lifecycles/{lifecycle_id}")
async def update_lifecycle(lifecycle_id: str, req: UpdateLifecycleRequest):
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM e0_lifecycles WHERE id=?", [lifecycle_id]).fetchone()
        if not row:
            raise HTTPException(404, f"lifecycle '{lifecycle_id}' not found")
        updates, params = [], []
        if req.name is not None:
            updates.append("name=?"); params.append(req.name)
        if req.description is not None:
            updates.append("description=?"); params.append(req.description)
        if req.enabled is not None:
            updates.append("enabled=?"); params.append(1 if req.enabled else 0)
        if not updates:
            return {"ok": True, "changed": False}
        updates.append("updated_at=?"); params.append(datetime.now().isoformat())
        params.append(lifecycle_id)
        conn.execute(f"UPDATE e0_lifecycles SET {', '.join(updates)} WHERE id=?", params)
        conn.commit()
        return {"ok": True, "id": lifecycle_id}
    finally:
        conn.close()


@router.delete("/lifecycles/{lifecycle_id}")
async def delete_lifecycle(lifecycle_id: str):
    conn = get_conn()
    try:
        row = conn.execute("SELECT id FROM e0_lifecycles WHERE id=?", [lifecycle_id]).fetchone()
        if not row:
            raise HTTPException(404, f"lifecycle '{lifecycle_id}' not found")
        conn.execute("DELETE FROM e0_lifecycles WHERE id=?", [lifecycle_id])
        conn.commit()
        logger.info(f"[E0] deleted lifecycle: {lifecycle_id}")
        return {"ok": True, "id": lifecycle_id}
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/lifecycles/{lifecycle_id}/state")
async def get_state(lifecycle_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM e0_lifecycles WHERE id=?", [lifecycle_id]).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"lifecycle '{lifecycle_id}' not found")
    r = dict(row)
    return {
        "id": r["id"], "state": r["state"],
        "state_meta": json.loads(r["state_meta"] or "{}"),
        "allowed_transitions": VALID_TRANSITIONS.get(r["state"], []),
        "updated_at": r["updated_at"],
    }


class TransitionRequest(BaseModel):
    to_state: str
    meta: dict = {}


@router.post("/lifecycles/{lifecycle_id}/state/transition")
async def transition_state(lifecycle_id: str, req: TransitionRequest):
    from modules.prvse.s.lifecycle import E0LifecycleManager
    try:
        row = E0LifecycleManager.transition(lifecycle_id, req.to_state)
        # persist meta if provided
        if req.meta:
            conn = get_conn()
            conn.execute(
                "UPDATE e0_lifecycles SET state_meta=? WHERE id=?",
                [json.dumps(req.meta), lifecycle_id],
            )
            conn.commit()
            conn.close()
        logger.info(f"[E0] {lifecycle_id}: → {req.to_state}")
        return {
            "ok": True, "id": lifecycle_id,
            "state": req.to_state,
            "allowed_transitions": VALID_TRANSITIONS.get(req.to_state, []),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PRVSE Components CRUD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/components")
async def list_components(
    lifecycle_id: Optional[str] = Query(None),
    layer: Optional[str] = Query(None),
):
    conn = get_conn()
    sql = "SELECT * FROM prvse_components WHERE 1=1"
    params: list = []
    if lifecycle_id:
        sql += " AND lifecycle_id=?"; params.append(lifecycle_id)
    if layer:
        sql += " AND layer=?"; params.append(layer.upper())
    sql += " ORDER BY lifecycle_id, layer, sub_id"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d.get("config") or "{}")
        result.append(d)
    return result


class CreateComponentRequest(BaseModel):
    lifecycle_id: str
    layer: str          # P|R|V|S|E
    sub_id: str
    name: str
    description: str = ""
    config: dict = {}


@router.post("/components", status_code=201)
async def create_component(req: CreateComponentRequest):
    layer = req.layer.upper()
    if layer not in ("P", "R", "V", "S", "E"):
        raise HTTPException(400, "layer must be one of P/R/V/S/E")
    if not req.lifecycle_id.strip() or not req.sub_id.strip() or not req.name.strip():
        raise HTTPException(400, "lifecycle_id, sub_id, name are required")

    conn = get_conn()
    try:
        lc = conn.execute("SELECT id FROM e0_lifecycles WHERE id=?", [req.lifecycle_id]).fetchone()
        if not lc:
            raise HTTPException(404, f"lifecycle '{req.lifecycle_id}' not found")

        cid = f"{req.lifecycle_id}.{layer}.{req.sub_id.strip()}"
        if conn.execute("SELECT id FROM prvse_components WHERE id=?", [cid]).fetchone():
            raise HTTPException(409, f"component '{cid}' already exists")

        now = datetime.now().isoformat()
        conn.execute(
            """INSERT INTO prvse_components
               (id, lifecycle_id, layer, sub_id, name, description, status, config, is_builtin, created_at, updated_at)
               VALUES (?,?,?,?,?,?,'inactive',?,0,?,?)""",
            [cid, req.lifecycle_id, layer, req.sub_id.strip(), req.name.strip(),
             req.description, json.dumps(req.config), now, now],
        )
        conn.commit()
        logger.info(f"[E0] created component: {cid}")
        return {"id": cid, "layer": layer, "sub_id": req.sub_id, "is_builtin": False}
    finally:
        conn.close()


class UpdateComponentRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    config: Optional[dict] = None


@router.patch("/components/{component_id:path}")
async def update_component(component_id: str, req: UpdateComponentRequest):
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM prvse_components WHERE id=?", [component_id]).fetchone()
        if not row:
            raise HTTPException(404, f"component '{component_id}' not found")

        updates, params = [], []
        if req.name is not None:
            updates.append("name=?"); params.append(req.name)
        if req.description is not None:
            updates.append("description=?"); params.append(req.description)
        if req.status is not None:
            if req.status not in ("active", "inactive", "error", "running"):
                raise HTTPException(400, "status must be active|inactive|error|running")
            updates.append("status=?"); params.append(req.status)
        if req.config is not None:
            updates.append("config=?"); params.append(json.dumps(req.config))
        if not updates:
            return {"ok": True, "changed": False}
        updates.append("updated_at=?"); params.append(datetime.now().isoformat())
        params.append(component_id)
        conn.execute(f"UPDATE prvse_components SET {', '.join(updates)} WHERE id=?", params)
        conn.commit()
        return {"ok": True, "id": component_id}
    finally:
        conn.close()


@router.delete("/components/{component_id:path}")
async def delete_component(component_id: str):
    conn = get_conn()
    try:
        row = conn.execute("SELECT id FROM prvse_components WHERE id=?", [component_id]).fetchone()
        if not row:
            raise HTTPException(404, f"component '{component_id}' not found")
        conn.execute("DELETE FROM prvse_components WHERE id=?", [component_id])
        conn.commit()
        logger.info(f"[E0] deleted component: {component_id}")
        return {"ok": True, "id": component_id}
    finally:
        conn.close()
