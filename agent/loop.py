"""
Main agent loop. Reads Graph from Egonetics, traverses nodes, executes them.
Handles backtracking, lifecycle state, 24h continuous operation.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from client.egonetics import egonetics
from agent.executor import NodeExecutor, NodeResult
from agent.backtrack import BacktrackManager
from store.db import create_feedback, get_pending_feedback
from config.settings import settings

logger = logging.getLogger(__name__)


class AgentLoop:

    def __init__(self, executor: NodeExecutor, emit=None):
        self.executor = executor
        self.backtrack = BacktrackManager()
        self._running = False
        self._current_task_id: Optional[str] = None
        # emit(event_type, data) callable for WebSocket broadcast; no-op if None
        self._emit = emit or (lambda e, d: asyncio.sleep(0))

    async def run_task(self, task_id: str, canvas_id: str):
        """Run agent loop for a specific task+canvas until completion or failure."""
        self._running = True
        self._current_task_id = task_id

        logger.info(f"Starting agent loop: task={task_id} canvas={canvas_id}")
        await egonetics.update_task(task_id, {"column_id": "in-progress"})

        try:
            await self._execute_graph(task_id, canvas_id)
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            await egonetics.update_task(task_id, {"column_id": "planned", "task_summary": f"Error: {e}"})
        finally:
            self._running = False

    async def _execute_graph(self, task_id: str, canvas_id: str):
        nodes = await egonetics.get_nodes(canvas_id)
        relations = await egonetics.get_relations(source_id=canvas_id)

        # Build adjacency: node_id → [child node_ids]
        adjacency = self._build_adjacency(nodes, relations)
        node_map = {n["id"]: n for n in nodes}

        # Find start node (no incoming edges, or node_kind == 'lifecycle' with action='start')
        start_nodes = self._find_start_nodes(nodes, relations)
        if not start_nodes:
            logger.warning("No start node found in graph")
            return

        context = {
            "task_id": task_id,
            "canvas_id": canvas_id,
            "accumulated_cost": {},
            "history": [],
            "variables": {}
        }

        # DFS traversal with backtracking
        for start_id in start_nodes:
            success = await self._traverse(start_id, node_map, adjacency, context)
            if success:
                await egonetics.update_task(task_id, {"column_id": "done"})
                return

        # All paths failed
        fb_id = create_feedback(
            task_id=task_id,
            feedback_type="failure_analysis",
            context={"canvas_id": canvas_id, "history": context["history"]},
            prompt="所有执行路径均失败，请分析根因并提供解决方案：",
            is_blocking=False
        )
        await egonetics.update_task(task_id, {
            "column_id": "review",
            "task_summary": f"执行失败，等待用户分析 (feedback: {fb_id})"
        })

    async def _traverse(self, node_id: str, node_map: dict, adjacency: dict, context: dict) -> bool:
        """Recursive DFS traversal with backtracking."""
        if node_id not in node_map:
            return False

        node = node_map[node_id]
        canvas_id = context["canvas_id"]

        # Update lifecycle state
        await egonetics.set_node_lifecycle(canvas_id, node_id, "running")
        await self._emit("node_start", {"node_id": node_id, "canvas_id": canvas_id,
                                        "node_kind": node.get("node_kind", "entity")})

        result: NodeResult = await self.executor.execute(node, context)

        if result.success:
            await egonetics.set_node_lifecycle(canvas_id, node_id, "success", result.cost)
            await self._emit("node_complete", {"node_id": node_id, "canvas_id": canvas_id,
                                               "cost": result.cost})
            context["history"].append({"node_id": node_id, "status": "success", "output": result.output})
            self._accumulate_cost(context, result.cost)

            # Determine next nodes
            if result.next_node_hint:
                children = [result.next_node_hint] if result.next_node_hint in node_map else []
            else:
                children = adjacency.get(node_id, [])

            if not children:
                return True  # Leaf node → success

            # Try each child
            for child_id in children:
                ok = await self._traverse(child_id, node_map, adjacency, context)
                if ok:
                    return True
                # Child failed → try next child (backtrack)
                await egonetics.set_node_lifecycle(canvas_id, child_id, "pending")

            # All children failed → this path fails
            return False

        else:
            error = result.error or ""

            # Handle waiting_human / low_confidence
            if "waiting_human:" in error or "low_confidence:" in error:
                fb_id = error.split(":")[-1]
                await egonetics.set_node_lifecycle(canvas_id, node_id, "waiting_human")
                # Pause and wait for feedback to be resolved
                resolved = await self._wait_for_feedback(fb_id, context["task_id"])
                if resolved:
                    # Retry node with user answer in context
                    context["variables"][f"human_answer_{node_id}"] = resolved
                    return await self._traverse(node_id, node_map, adjacency, context)
                return False

            await egonetics.set_node_lifecycle(canvas_id, node_id, "failed")
            await self._emit("node_failed", {"node_id": node_id, "canvas_id": canvas_id, "error": error})
            context["history"].append({"node_id": node_id, "status": "failed", "error": error})
            return False

    async def _wait_for_feedback(self, feedback_id: str, task_id: str,
                                  poll_interval: int = 10, max_wait: int = 86400) -> Optional[str]:
        """Poll for feedback resolution. Returns user_response or None on timeout."""
        elapsed = 0
        while elapsed < max_wait:
            pending = get_pending_feedback(task_id)
            fb = next((f for f in pending if f["id"] == feedback_id), None)
            if fb is None:
                # Not in pending → resolved
                from store.db import get_conn, row_to_dict
                conn = get_conn()
                row = conn.execute("SELECT * FROM user_feedback WHERE id=?", [feedback_id]).fetchone()
                conn.close()
                if row:
                    r = row_to_dict(row)
                    return r.get("user_response")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        return None

    def _build_adjacency(self, nodes: list, relations: list) -> dict:
        adj = {}
        for r in relations:
            src = r.get("source_id")
            tgt = r.get("target_id")
            if src and tgt:
                adj.setdefault(src, []).append(tgt)
        return adj

    def _find_start_nodes(self, nodes: list, relations: list) -> list:
        has_incoming = {r["target_id"] for r in relations if r.get("target_id")}
        starts = [n["id"] for n in nodes if n["id"] not in has_incoming]
        return starts

    def _accumulate_cost(self, context: dict, cost: dict):
        acc = context["accumulated_cost"]
        for k, v in cost.items():
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0) + v
