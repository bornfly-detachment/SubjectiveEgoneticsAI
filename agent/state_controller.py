"""
State Controller — 24/7 任务队列轮询器
每隔 POLL_INTERVAL 秒从 execution_queue 拉取待执行任务，分发给 AgentLoop。
状态转换时自动打 S 层标签（tag_trees），聚合 resource_cost，计算 V score。
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from agent.loop import AgentLoop
from client.egonetics import egonetics
from modules.action import ActionModule
from modules.judge import JudgeModule
from store.db import get_conn

logger = logging.getLogger(__name__)

POLL_INTERVAL  = 5   # 秒
MAX_CONCURRENT = 3

# ── S 层状态机标签映射 ──────────────────────────────────────────────────────
# tag IDs 来自 Egonetics tag_trees (tag-s-n-sm-*)
STATE_TAG: dict[str, str] = {
    "running":  "tag-s-n-sm-build",   # 构建中
    "done_pos": "tag-s-n-sm-pos",     # 正反馈迭代（v_score >= 0.7）
    "done_neg": "tag-s-n-sm-neg",     # 负反馈预警（v_score < 0.7）
    "done_arc": "tag-s-n-sm-arch",    # 归档（无 v_score 时默认）
    "failed":   "tag-s-n-sm-bug",     # Bug 挂起
    "blocked":  "tag-s-n-sm-wait",    # 等待指令挂起
}

V_POSITIVE_THRESHOLD = 0.7  # V score 阈值


class StateController:
    def __init__(self):
        self._running = False
        self._paused  = False
        self._last_poll_at: Optional[str] = None
        self._active: dict[str, asyncio.Task] = {}  # queue_id → asyncio.Task

    # ── 生命周期 ──────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        logger.info("StateController started (poll=%ds, max=%d)", POLL_INTERVAL, MAX_CONCURRENT)
        while self._running:
            try:
                if not self._paused:
                    await self._poll()
                self._last_poll_at = datetime.now().isoformat()
            except Exception as e:
                logger.error("StateController poll error: %s", e, exc_info=True)
            await asyncio.sleep(POLL_INTERVAL)

    async def stop(self):
        self._running = False
        for t in self._active.values():
            t.cancel()
        logger.info("StateController stopped")

    def pause(self):
        self._paused = True
        logger.info("StateController paused")

    def resume(self):
        self._paused = False
        logger.info("StateController resumed")

    # ── 轮询逻辑（只处理 layer='local' 任务）────────────────────────────

    async def _poll(self):
        if len(self._active) >= MAX_CONCURRENT:
            return
        conn = get_conn()
        # 只拉取局部层任务；若 layer 字段不存在（旧数据），也包含进来
        try:
            rows = conn.execute(
                """SELECT * FROM execution_queue
                   WHERE state='pending' AND (layer='local' OR layer IS NULL)
                   ORDER BY sort_order, created_at"""
            ).fetchall()
        except Exception:
            # 迁移前的旧表没有 layer 字段，回退到原逻辑
            rows = conn.execute(
                "SELECT * FROM execution_queue WHERE state='pending' ORDER BY sort_order, created_at"
            ).fetchall()
        conn.close()
        for task in [dict(r) for r in rows]:
            if len(self._active) >= MAX_CONCURRENT:
                break
            if task["id"] not in self._active:
                await self._dispatch(task)

    # ── 状态写入 ──────────────────────────────────────────────────────────

    def _queue_patch(self, queue_id: str, **kwargs):
        fields = [f"{k}=?" for k in kwargs]
        values = list(kwargs.values()) + [queue_id]
        conn = get_conn()
        conn.execute(f"UPDATE execution_queue SET {', '.join(fields)} WHERE id=?", values)
        conn.commit()
        conn.close()

    def _set_state_tag(self, queue_id: str, tag_key: str):
        tag_id = STATE_TAG.get(tag_key)
        if tag_id:
            self._queue_patch(queue_id, state_tags=json.dumps([tag_id]))

    def _aggregate_cost(self, loop_task_id: str) -> dict:
        """从 trajectories 聚合该任务的资源消耗"""
        conn = get_conn()
        rows = conn.execute(
            "SELECT cost_vector FROM trajectories WHERE task_id=?",
            [loop_task_id]
        ).fetchall()
        conn.close()
        total = {"token_input": 0, "token_output": 0, "time_ms": 0}
        for r in rows:
            try:
                cv = json.loads(r["cost_vector"] or "{}")
                total["token_input"]  += cv.get("token_input", 0) or 0
                total["token_output"] += cv.get("token_output", 0) or 0
                total["time_ms"]      += cv.get("time_ms", 0) or 0
            except Exception:
                pass
        return total

    def _compute_v_score(self, loop_task_id: str) -> Optional[float]:
        """从 trajectories.reward 计算平均 V score"""
        conn = get_conn()
        rows = conn.execute(
            "SELECT reward FROM trajectories WHERE task_id=? AND reward IS NOT NULL",
            [loop_task_id]
        ).fetchall()
        conn.close()
        rewards = [r["reward"] for r in rows if r["reward"] is not None]
        if not rewards:
            return None
        return round(sum(rewards) / len(rewards), 3)

    # ── 任务分发 ──────────────────────────────────────────────────────────

    async def _dispatch(self, task: dict):
        queue_id    = task["id"]
        task_desc   = task.get("description") or queue_id
        loop_task_id = task.get("task_ref_id") or queue_id

        # 查找或创建 execution canvas
        try:
            canvases = await egonetics.get_execution_canvases(loop_task_id)
            canvas_id = canvases[0]["id"] if canvases else (
                await egonetics.create_execution_canvas(loop_task_id, f"执行: {task_desc}")
            )["id"]
        except Exception as e:
            logger.warning("Canvas lookup failed for %s: %s", loop_task_id, e)
            canvas_id = loop_task_id  # fallback

        # 标记 running + 打 S 标签
        self._queue_patch(
            queue_id,
            state="running",
            assigned_node="local",
            started_at=datetime.now().isoformat(),
        )
        self._set_state_tag(queue_id, "running")

        loop = AgentLoop(action_module=ActionModule(), judge_module=JudgeModule())

        async def run_and_cleanup():
            try:
                await loop.run(loop_task_id, canvas_id)
                # 聚合 cost + 计算 V
                cost    = self._aggregate_cost(loop_task_id)
                v_score = self._compute_v_score(loop_task_id)
                tag_key = (
                    "done_pos" if v_score is not None and v_score >= V_POSITIVE_THRESHOLD
                    else "done_neg" if v_score is not None
                    else "done_arc"
                )
                self._queue_patch(
                    queue_id,
                    state="done",
                    completed_at=datetime.now().isoformat(),
                    v_score=v_score,
                    resource_cost=json.dumps(cost),
                )
                self._set_state_tag(queue_id, tag_key)

            except asyncio.CancelledError:
                self._queue_patch(queue_id, state="pending", assigned_node=None)
                self._queue_patch(queue_id, state_tags=json.dumps([]))

            except Exception as e:
                logger.error("Task %s failed: %s", queue_id, e, exc_info=True)
                cost = self._aggregate_cost(loop_task_id)
                self._queue_patch(
                    queue_id,
                    state="failed",
                    error_msg=str(e),
                    completed_at=datetime.now().isoformat(),
                    resource_cost=json.dumps(cost),
                )
                self._set_state_tag(queue_id, "failed")

            finally:
                self._active.pop(queue_id, None)

        t = asyncio.create_task(run_and_cleanup(), name=f"task-{queue_id}")
        self._active[queue_id] = t
        logger.info("Dispatched queue=%s loop_task=%s canvas=%s", queue_id, loop_task_id, canvas_id)

    # ── 只读状态（供 API 路由查询）─────────────────────────────────────────

    def status(self) -> dict:
        return {
            "is_running":    self._running,
            "is_paused":     self._paused,
            "active_tasks":  list(self._active.keys()),
            "active_count":  len(self._active),
            "poll_interval": POLL_INTERVAL,
            "max_concurrent": MAX_CONCURRENT,
            "last_poll_at":  self._last_poll_at,
        }


state_controller = StateController()
