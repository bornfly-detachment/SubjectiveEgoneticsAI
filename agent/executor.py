"""
Node executor: dispatches execution based on node_kind.
7 node types: llm_call, tool_call, local_judge, rule_branch, human_gate, lifecycle, cost_tracker
"""
import asyncio
import time
import psutil
from typing import Any
from datetime import datetime

from store.db import save_trajectory, complete_trajectory, fail_trajectory, create_feedback, gen_id
from config.settings import settings


class NodeResult:
    def __init__(self, success: bool, output: Any = None, error: str = None,
                 cost: dict = None, next_node_hint: str = None):
        self.success = success
        self.output = output
        self.error = error
        self.cost = cost or {}
        self.next_node_hint = next_node_hint  # for rule_branch decisions


class NodeExecutor:

    def __init__(self, action_module, judge_module, feedback_notifier):
        self.action = action_module
        self.judge = judge_module
        self.notify = feedback_notifier  # callable(feedback_id, task_id, msg)
        self._loop_counter: dict[str, int] = {}  # node_id → visit count

    async def execute(self, node: dict, context: dict) -> NodeResult:
        kind = node.get("node_kind", "entity")
        node_id = node["id"]
        task_id = context.get("task_id", "unknown")
        canvas_id = context.get("canvas_id", "")

        # Loop detection
        self._loop_counter[node_id] = self._loop_counter.get(node_id, 0) + 1
        if self._loop_counter[node_id] > settings.max_loop_detection_count:
            return NodeResult(False, error=f"Loop detected on node {node_id}", cost={})

        exec_config = node.get("exec_config") or {}
        if isinstance(exec_config, str):
            import json
            exec_config = json.loads(exec_config) if exec_config else {}

        traj_id = save_trajectory(task_id, canvas_id, node_id, kind, {
            "node": node, "context": context
        })

        t0 = time.time()
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            if kind == "llm_call":
                result = await self._exec_llm_call(exec_config, context)
            elif kind == "tool_call":
                result = await self._exec_tool_call(exec_config, context)
            elif kind == "local_judge":
                result = await self._exec_local_judge(exec_config, context)
            elif kind == "rule_branch":
                result = self._exec_rule_branch(exec_config, context)
            elif kind == "human_gate":
                result = await self._exec_human_gate(exec_config, context, task_id, traj_id)
            elif kind == "lifecycle":
                result = self._exec_lifecycle(exec_config, context)
            elif kind == "cost_tracker":
                result = self._exec_cost_tracker(exec_config, context)
            else:
                # Unknown / entity node — pass through
                result = NodeResult(True, output={"entity_id": node.get("entity_id")})

            elapsed_ms = int((time.time() - t0) * 1000)
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            cost = {
                "time_ms": elapsed_ms,
                "memory_mb": round(mem_after - mem_before, 2),
                **result.cost
            }
            result.cost = cost

            if result.success:
                complete_trajectory(traj_id, result.output or {}, cost)
            else:
                fail_trajectory(traj_id, result.error or "unknown error")

            return result

        except asyncio.TimeoutError:
            fail_trajectory(traj_id, "Execution timeout", "timeout")
            return NodeResult(False, error="Timeout", cost={})
        except Exception as e:
            fail_trajectory(traj_id, str(e), "capability")
            return NodeResult(False, error=str(e), cost={})

    async def _exec_llm_call(self, cfg: dict, ctx: dict) -> NodeResult:
        prompt = cfg.get("prompt", "")
        system = cfg.get("system", "")
        model = cfg.get("model", settings.default_llm_model)
        budget_tokens = cfg.get("budget_tokens", 4000)

        result = await self.action.llm_call(
            prompt=prompt,
            system=system,
            context=ctx,
            model=model,
            max_tokens=budget_tokens
        )
        return NodeResult(
            success=True,
            output=result["content"],
            cost={
                "token_input": result.get("input_tokens", 0),
                "token_output": result.get("output_tokens", 0)
            }
        )

    async def _exec_tool_call(self, cfg: dict, ctx: dict) -> NodeResult:
        tool_name = cfg.get("tool", "")
        tool_args = cfg.get("args", {})
        result = await self.action.tool_call(tool_name, tool_args, ctx)
        return NodeResult(success=result.get("ok", False), output=result)

    async def _exec_local_judge(self, cfg: dict, ctx: dict) -> NodeResult:
        question = cfg.get("question", "")
        threshold = cfg.get("confidence_threshold", settings.judge_confidence_threshold)

        judgment = await self.judge.judge(question=question, context=ctx)
        confident = judgment.get("confidence", 0) >= threshold

        if not confident:
            # Ask user via decision_query feedback
            fb_id = create_feedback(
                task_id=ctx.get("task_id", ""),
                feedback_type="decision_query",
                context={"question": question, "model_answer": judgment, "node_config": cfg},
                prompt=f"模型不确定：{question}\n模型建议：{judgment.get('answer', '?')}\n置信度：{judgment.get('confidence', 0):.2f}\n请给出你的判断：",
                is_blocking=True
            )
            return NodeResult(
                False,
                error=f"low_confidence:feedback:{fb_id}",
                output={"feedback_id": fb_id, "model_judgment": judgment}
            )

        return NodeResult(True, output=judgment)

    def _exec_rule_branch(self, cfg: dict, ctx: dict) -> NodeResult:
        """Evaluate a Python expression rule to decide next path."""
        condition = cfg.get("condition", "True")
        branches = cfg.get("branches", {})  # {"true": node_id, "false": node_id}

        try:
            result = bool(eval(condition, {"ctx": ctx, "__builtins__": {}}))
            next_node = branches.get("true" if result else "false")
            return NodeResult(True, output={"condition_result": result}, next_node_hint=next_node)
        except Exception as e:
            return NodeResult(False, error=f"Rule eval error: {e}")

    async def _exec_human_gate(self, cfg: dict, ctx: dict, task_id: str, traj_id: str) -> NodeResult:
        is_blocking = cfg.get("blocking", True)
        prompt_text = cfg.get("prompt", "需要你的确认才能继续")

        fb_id = create_feedback(
            task_id=task_id,
            feedback_type="decision_query",
            context={"config": cfg, "context": ctx},
            prompt=prompt_text,
            is_blocking=is_blocking
        )

        if is_blocking:
            # Notify user and wait (poll every 5s, max 24h)
            await self.notify(fb_id, task_id, prompt_text)
            return NodeResult(
                False,
                error=f"waiting_human:feedback:{fb_id}",
                output={"feedback_id": fb_id}
            )
        else:
            # Non-blocking: queue it and continue
            await self.notify(fb_id, task_id, prompt_text)
            return NodeResult(True, output={"feedback_id": fb_id, "queued": True})

    def _exec_lifecycle(self, cfg: dict, ctx: dict) -> NodeResult:
        action = cfg.get("action", "checkpoint")
        return NodeResult(True, output={"lifecycle_action": action, "timestamp": datetime.now().isoformat()})

    def _exec_cost_tracker(self, cfg: dict, ctx: dict) -> NodeResult:
        budget = cfg.get("budget", {})
        return NodeResult(True, output={"budget_snapshot": budget, "context_cost": ctx.get("accumulated_cost", {})})
