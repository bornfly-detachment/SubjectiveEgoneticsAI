"""
VRegistry — V 层奖励函数注册表（AOP 核心）

两类函数统一管理：
  is_builtin=1  系统函数：Python @reward_function 装饰，逻辑在代码里；启动时 upsert 进 DB
  is_builtin=0  用户函数：expression 字段存储 Python 表达式，eval(expression, {"ctx": ctx}) 计算分数

CRUD 全部走 DB，内存 VRegistry 是 DB 的热缓存。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from modules.prvse.v.context import ExecutionContext

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class VFunction:
    id:          str
    name:        str
    description: str
    unit:        str
    weight:      float
    trigger:     str
    enabled:     bool
    is_builtin:  bool
    fn:          Optional[Callable] = field(default=None, repr=False)   # 系统函数
    expression:  Optional[str]      = None                              # 用户函数


@dataclass
class RewardResult:
    total_reward: float
    functions:    dict[str, dict]   # name → {score, weight, weighted, error?}


# ── 注册表 ────────────────────────────────────────────────────────────────────

class VRegistry:
    _functions: dict[str, VFunction] = {}

    # ── AOP 装饰器 ────────────────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name:        str,
        description: str,
        unit:        str   = "score",
        weight:      float = 1.0,
        trigger:     str   = "any",
    ):
        def decorator(fn: Callable) -> Callable:
            cls._functions[name] = VFunction(
                id=name, name=name, description=description,
                unit=unit, weight=weight, trigger=trigger,
                enabled=True, is_builtin=True, fn=fn,
            )
            logger.debug(f"[VRegistry] registered builtin: {name}")
            return fn
        return decorator

    # ── 启动初始化：DB 同步 ───────────────────────────────────────────────────

    @classmethod
    def sync_builtins_to_db(cls):
        """
        把 @reward_function 装饰的系统函数 upsert 进 v_functions 表。
        已有行：保留用户改过的 weight / description / enabled。
        新行：写入默认值。
        """
        try:
            from store.db import get_conn
            conn = get_conn()
            now = datetime.now().isoformat()
            for vf in cls._functions.values():
                if not vf.is_builtin:
                    continue
                existing = conn.execute(
                    "SELECT weight, description, enabled FROM v_functions WHERE id=?", [vf.id]
                ).fetchone()
                if existing:
                    # 保留用户修改，仅更新不可变字段
                    vf.weight      = existing["weight"]
                    vf.description = existing["description"] or vf.description
                    vf.enabled     = bool(existing["enabled"])
                    conn.execute(
                        "UPDATE v_functions SET unit=?, trigger=?, is_builtin=1, updated_at=? WHERE id=?",
                        [vf.unit, vf.trigger, now, vf.id]
                    )
                else:
                    conn.execute(
                        """INSERT INTO v_functions
                           (id, name, description, unit, weight, trigger, expression, enabled, is_builtin, created_at, updated_at)
                           VALUES (?,?,?,?,?,?,NULL,1,1,?,?)""",
                        [vf.id, vf.name, vf.description, vf.unit, vf.weight,
                         vf.trigger, now, now]
                    )
            conn.commit()
            conn.close()
            logger.info(f"[VRegistry] synced {len(cls._functions)} builtins to DB")
        except Exception as e:
            logger.warning(f"[VRegistry] sync_builtins_to_db failed: {e}")

    @classmethod
    def load_user_functions_from_db(cls):
        """从 DB 加载 is_builtin=0 的用户自定义函数到内存。"""
        try:
            from store.db import get_conn
            conn = get_conn()
            rows = conn.execute(
                "SELECT * FROM v_functions WHERE is_builtin=0 AND enabled=1"
            ).fetchall()
            conn.close()
            for row in rows:
                r = dict(row)
                cls._functions[r["id"]] = VFunction(
                    id=r["id"], name=r["name"], description=r["description"],
                    unit=r["unit"], weight=r["weight"], trigger=r["trigger"],
                    enabled=True, is_builtin=False, expression=r["expression"],
                )
            logger.info(f"[VRegistry] loaded {len(rows)} user functions from DB")
        except Exception as e:
            logger.warning(f"[VRegistry] load_user_functions_from_db failed: {e}")

    @classmethod
    def reload_all_from_db(cls):
        """重新从 DB 同步所有函数的状态（weight / enabled / description）。"""
        try:
            from store.db import get_conn
            conn = get_conn()
            rows = conn.execute("SELECT * FROM v_functions").fetchall()
            conn.close()
            for row in rows:
                r = dict(row)
                fid = r["id"]
                if fid in cls._functions:
                    cls._functions[fid].weight      = r["weight"]
                    cls._functions[fid].enabled     = bool(r["enabled"])
                    cls._functions[fid].description = r["description"]
                    cls._functions[fid].trigger     = r["trigger"]
                    if not cls._functions[fid].is_builtin:
                        cls._functions[fid].expression = r["expression"]
                else:
                    # 用户函数：直接加进来
                    if not r["is_builtin"]:
                        cls._functions[fid] = VFunction(
                            id=fid, name=r["name"], description=r["description"],
                            unit=r["unit"], weight=r["weight"], trigger=r["trigger"],
                            enabled=bool(r["enabled"]), is_builtin=False,
                            expression=r["expression"],
                        )
        except Exception as e:
            logger.warning(f"[VRegistry] reload_all_from_db failed: {e}")

    # ── 计算 ──────────────────────────────────────────────────────────────────

    @classmethod
    def compute(cls, ctx: ExecutionContext) -> RewardResult:
        results: dict[str, dict] = {}
        total = 0.0

        for name, vf in cls._functions.items():
            if not vf.enabled:
                continue
            if vf.trigger != "any" and vf.trigger != ctx.node_kind:
                continue
            try:
                if vf.is_builtin and vf.fn:
                    score = float(vf.fn(ctx))
                elif vf.expression:
                    score = float(eval(
                        vf.expression,
                        {"ctx": ctx, "min": min, "max": max, "abs": abs, "__builtins__": {}},
                    ))
                else:
                    continue
                weighted = score * vf.weight
                results[name] = {"score": score, "weight": vf.weight, "weighted": weighted}
                total += weighted
            except Exception as e:
                logger.warning(f"[VRegistry] {name} raised: {e}")
                results[name] = {"score": 0.0, "weight": vf.weight, "weighted": 0.0, "error": str(e)}

        return RewardResult(total_reward=round(total, 4), functions=results)

    # ── 内存热更新 ────────────────────────────────────────────────────────────

    @classmethod
    def set_weight(cls, name: str, weight: float) -> bool:
        if name not in cls._functions:
            return False
        cls._functions[name].weight = weight
        return True

    @classmethod
    def add_user_function(cls, vf: VFunction):
        cls._functions[vf.id] = vf

    @classmethod
    def remove_function(cls, fid: str) -> bool:
        if fid not in cls._functions:
            return False
        del cls._functions[fid]
        return True

    # ── 元信息 ────────────────────────────────────────────────────────────────

    @classmethod
    def list_functions(cls) -> list[dict]:
        return [
            {
                "id":          vf.id,
                "name":        vf.name,
                "description": vf.description,
                "unit":        vf.unit,
                "weight":      vf.weight,
                "trigger":     vf.trigger,
                "enabled":     vf.enabled,
                "is_builtin":  vf.is_builtin,
                "expression":  vf.expression,
            }
            for vf in cls._functions.values()
        ]


# ── 公开别名 ──────────────────────────────────────────────────────────────────

def reward_function(name: str, description: str, unit: str = "score",
                    weight: float = 1.0, trigger: str = "any"):
    return VRegistry.register(name=name, description=description,
                               unit=unit, weight=weight, trigger=trigger)


def compute_reward(ctx: ExecutionContext) -> RewardResult:
    return VRegistry.compute(ctx)
