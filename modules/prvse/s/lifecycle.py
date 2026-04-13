"""
E0 Lifecycle State Machine — S 层核心

States: IDLE → OBSERVING → REFLECTING → TRAINING → VALIDATING → ACTIVATING → IDLE

CRUD 全部走 DB，内存状态是 DB 的缓存快照。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

VALID_STATES = ["IDLE", "OBSERVING", "REFLECTING", "TRAINING", "VALIDATING", "ACTIVATING"]

VALID_TRANSITIONS: dict[str, list[str]] = {
    "IDLE":       ["OBSERVING"],
    "OBSERVING":  ["REFLECTING", "IDLE"],
    "REFLECTING": ["TRAINING", "IDLE"],
    "TRAINING":   ["VALIDATING", "IDLE"],
    "VALIDATING": ["ACTIVATING", "REFLECTING"],
    "ACTIVATING": ["IDLE"],
}

# ── 内置生命周期 ───────────────────────────────────────────────────────────────

BUILTIN_LIFECYCLES = [
    {"id": "e0",    "name": "E0 全局",   "description": "Always-running cybernetic self-evolution lifecycle"},
    {"id": "task",  "name": "Task",      "description": "Per-task execution lifecycle"},
    {"id": "agent", "name": "Agent",     "description": "Agent instance lifecycle"},
    {"id": "model", "name": "Model",     "description": "Local model training/serving lifecycle"},
]

# ── 内置 PRVSE 组件（仅 e0 生命周期） ─────────────────────────────────────────

BUILTIN_COMPONENTS = [
    # P 感知层
    {"lifecycle_id": "e0", "layer": "P", "sub_id": "observe",   "name": "观察",   "description": "收集原始数据：trajectories、feedback、user actions"},
    {"lifecycle_id": "e0", "layer": "P", "sub_id": "classify",  "name": "分类",   "description": "对观察到的事件按类型/优先级分类"},
    {"lifecycle_id": "e0", "layer": "P", "sub_id": "detect",    "name": "检测",   "description": "检测异常、失败模式、重复错误"},
    {"lifecycle_id": "e0", "layer": "P", "sub_id": "compress",  "name": "压缩",   "description": "将原始观察压缩为结构化摘要"},
    # R 关系层
    {"lifecycle_id": "e0", "layer": "R", "sub_id": "entity",    "name": "实体",   "description": "识别关键实体：task/agent/model/rule"},
    {"lifecycle_id": "e0", "layer": "R", "sub_id": "link",      "name": "链接",   "description": "建立实体间因果/依赖关系"},
    {"lifecycle_id": "e0", "layer": "R", "sub_id": "infer",     "name": "推断",   "description": "从关系图推断隐含规律"},
    {"lifecycle_id": "e0", "layer": "R", "sub_id": "graph",     "name": "图谱",   "description": "维护知识图谱状态"},
    # V 价值层
    {"lifecycle_id": "e0", "layer": "V", "sub_id": "local",     "name": "局部价值", "description": "单次执行局部收益 [-1,1]"},
    {"lifecycle_id": "e0", "layer": "V", "sub_id": "global",    "name": "全局价值", "description": "对整体目标的贡献 [-1,1]"},
    {"lifecycle_id": "e0", "layer": "V", "sub_id": "now",       "name": "当下价值", "description": "立即收益 [0,1]"},
    {"lifecycle_id": "e0", "layer": "V", "sub_id": "future",    "name": "未来价值", "description": "长期潜力 [0,1]"},
    {"lifecycle_id": "e0", "layer": "V", "sub_id": "certainty", "name": "确定性",   "description": "价值判断的置信度 [0,1]"},
    # S 状态层
    {"lifecycle_id": "e0", "layer": "S", "sub_id": "define",     "name": "状态定义",  "description": "定义系统合法状态集合"},
    {"lifecycle_id": "e0", "layer": "S", "sub_id": "transition", "name": "状态转换",  "description": "驱动状态机流转"},
    {"lifecycle_id": "e0", "layer": "S", "sub_id": "lifecycle",  "name": "生命周期",  "description": "E0/Task/Agent/Model 生命周期管理"},
    # E 进化层
    {"lifecycle_id": "e0", "layer": "E", "sub_id": "diff",      "name": "Diff",     "description": "AI输出 vs ground_truth 的差距收集"},
    {"lifecycle_id": "e0", "layer": "E", "sub_id": "trigger",   "name": "触发条件", "description": "训练触发规则（diff阈值/数量/时间）"},
    {"lifecycle_id": "e0", "layer": "E", "sub_id": "train",     "name": "训练",     "description": "执行 GRPO/SFT 训练"},
    {"lifecycle_id": "e0", "layer": "E", "sub_id": "validate",  "name": "验证",     "description": "验证新模型性能"},
    {"lifecycle_id": "e0", "layer": "E", "sub_id": "activate",  "name": "激活",     "description": "A/B 测试通过后切换活跃模型"},
]


class E0LifecycleManager:
    """启动时同步内置数据到 DB，运行时所有读写走 DB。"""

    @classmethod
    def sync_builtins_to_db(cls):
        try:
            from store.db import get_conn
            conn = get_conn()
            now = datetime.now().isoformat()

            for lc in BUILTIN_LIFECYCLES:
                exists = conn.execute(
                    "SELECT id FROM e0_lifecycles WHERE id=?", [lc["id"]]
                ).fetchone()
                if not exists:
                    conn.execute(
                        """INSERT INTO e0_lifecycles
                           (id, name, description, state, state_meta, enabled, is_builtin, created_at, updated_at)
                           VALUES (?,?,?,'IDLE','{}',1,1,?,?)""",
                        [lc["id"], lc["name"], lc["description"], now, now],
                    )

            for comp in BUILTIN_COMPONENTS:
                cid = f"{comp['lifecycle_id']}.{comp['layer']}.{comp['sub_id']}"
                exists = conn.execute(
                    "SELECT id FROM prvse_components WHERE id=?", [cid]
                ).fetchone()
                if not exists:
                    conn.execute(
                        """INSERT INTO prvse_components
                           (id, lifecycle_id, layer, sub_id, name, description, status, config, is_builtin, created_at, updated_at)
                           VALUES (?,?,?,?,?,?,'inactive','{}',1,?,?)""",
                        [cid, comp["lifecycle_id"], comp["layer"], comp["sub_id"],
                         comp["name"], comp["description"], now, now],
                    )

            conn.commit()
            conn.close()
            logger.info(f"[E0] synced {len(BUILTIN_LIFECYCLES)} lifecycles, {len(BUILTIN_COMPONENTS)} components")
        except Exception as e:
            logger.warning(f"[E0] sync_builtins_to_db failed: {e}")

    @classmethod
    def transition(cls, lifecycle_id: str, to_state: str) -> dict:
        """Validate and execute a state transition. Returns updated lifecycle row."""
        from store.db import get_conn
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM e0_lifecycles WHERE id=?", [lifecycle_id]
            ).fetchone()
            if not row:
                raise ValueError(f"lifecycle '{lifecycle_id}' not found")

            row = dict(row)
            from_state = row["state"]

            if to_state not in VALID_STATES:
                raise ValueError(f"unknown state '{to_state}'")
            if to_state not in VALID_TRANSITIONS.get(from_state, []):
                raise ValueError(
                    f"invalid transition {from_state} → {to_state}; "
                    f"allowed: {VALID_TRANSITIONS.get(from_state, [])}"
                )

            now = datetime.now().isoformat()
            conn.execute(
                "UPDATE e0_lifecycles SET state=?, updated_at=? WHERE id=?",
                [to_state, now, lifecycle_id],
            )
            conn.commit()
            row["state"] = to_state
            row["updated_at"] = now
            return row
        finally:
            conn.close()
