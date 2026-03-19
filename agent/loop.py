"""
Agent loop — LangGraph-based execution with checkpoint/resume.

run()    → start fresh or continue from existing checkpoint
resume() → inject human feedback into an interrupted graph (Command(resume=...))
"""
import asyncio
import logging
from typing import Optional, Callable

from langgraph.types import Command

from agent.graph import build_graph, ExecState
from client.egonetics import egonetics
from store.db import resolve_feedback
from config.settings import settings

logger = logging.getLogger(__name__)


class AgentLoop:

    def __init__(self, action_module, judge_module, emit_fn: Optional[Callable] = None):
        self.action = action_module
        self.judge = judge_module
        # emit_fn: async (event_type: str, data: dict) → None
        self._emit = emit_fn or (lambda e, d: asyncio.sleep(0))
        self._compiled = None      # compiled LangGraph (held in memory for resume)
        self._checkpointer = None
        self._task_id: Optional[str] = None
        self._canvas_id: Optional[str] = None

    # ── Public API ───────────────────────────────────────────────────────────

    async def run(self, task_id: str, canvas_id: str):
        """
        Start or resume a task.
        If a checkpoint exists for task_id, LangGraph auto-resumes from last saved state.
        """
        self._task_id = task_id
        self._canvas_id = canvas_id

        await egonetics.update_task(task_id, {"column_id": "in-progress"})

        compiled, checkpointer, _ = await build_graph(
            canvas_id, self.action, self.judge, self._emit
        )
        self._compiled = compiled
        self._checkpointer = checkpointer

        config = {"configurable": {"thread_id": task_id}}
        initial = ExecState(
            task_id=task_id,
            canvas_id=canvas_id,
            outputs={},
            variables={},
            trajectory=[],
            failed_nodes=[],
        )

        try:
            await self._stream(compiled, initial, config, task_id, canvas_id)
        except Exception as e:
            logger.error(f"AgentLoop.run error task={task_id}: {e}", exc_info=True)
            await egonetics.update_task(task_id, {
                "column_id": "planned",
                "task_summary": f"执行异常: {e}",
            })
            raise

    async def resume(self, task_id: str, canvas_id: str, feedback_value: str):
        """
        Resume an interrupted graph with human feedback.
        Rebuilds the graph if not in memory.
        """
        if self._compiled is None or self._task_id != task_id:
            compiled, checkpointer, _ = await build_graph(
                canvas_id, self.action, self.judge, self._emit
            )
            self._compiled = compiled
            self._checkpointer = checkpointer
            self._task_id = task_id
            self._canvas_id = canvas_id

        config = {"configurable": {"thread_id": task_id}}
        cmd = Command(resume=feedback_value)

        try:
            await self._stream(self._compiled, cmd, config, task_id, canvas_id)
        except Exception as e:
            logger.error(f"AgentLoop.resume error task={task_id}: {e}", exc_info=True)
            raise

    # ── Streaming ────────────────────────────────────────────────────────────

    async def _stream(self, compiled, input_or_cmd, config: dict, task_id: str, canvas_id: str):
        """
        Stream LangGraph events and forward to WebSocket + Egonetics.
        Handles node lifecycle, human_gate interrupts, and completion.
        """
        done = False
        async for event in compiled.astream_events(input_or_cmd, config, version="v2"):
            etype = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            # ── Node-level events ──
            if etype == "on_chain_start" and name not in ("LangGraph", "__start__"):
                await self._emit("node_start", {
                    "node_id": name, "canvas_id": canvas_id,
                })

            elif etype == "on_chain_end" and name not in ("LangGraph", "__end__"):
                output = data.get("output", {})
                failed = output.get("failed_nodes", []) if isinstance(output, dict) else []
                if name in (failed or []):
                    await self._emit("node_failed", {"node_id": name, "canvas_id": canvas_id})
                else:
                    await self._emit("node_complete", {"node_id": name, "canvas_id": canvas_id})

            # ── Interrupt (human_gate) ──
            elif etype == "on_interrupt":
                interrupt_data = data.get("interrupt_data") or data
                await self._emit("human_gate", {
                    "canvas_id": canvas_id,
                    **interrupt_data,
                })

            # ── Graph completion ──
            elif etype == "on_chain_end" and name == "LangGraph":
                done = True

        if done:
            final_state = compiled.get_state(config)
            failed = final_state.values.get("failed_nodes", []) if final_state.values else []
            if failed:
                await egonetics.update_task(task_id, {
                    "column_id": "review",
                    "task_summary": f"部分节点失败: {failed}",
                })
                await self._emit("task_failed", {"failed_nodes": failed})
            else:
                await egonetics.update_task(task_id, {"column_id": "done"})
                await self._emit("task_done", {"message": "任务执行完成"})
