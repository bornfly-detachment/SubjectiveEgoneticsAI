"""
LangGraph node function factories.
Each factory returns an async callable (state: ExecState) -> dict.
"""
import json
import logging
from typing import Callable, Any

from langgraph.types import interrupt

from store.db import save_trajectory, complete_trajectory, fail_trajectory, create_feedback, gen_id
from config.settings import settings

logger = logging.getLogger(__name__)


# ── State reducers ──────────────────────────────────────────────────────────

def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


# ── Node factories ──────────────────────────────────────────────────────────

def make_node_fn(
    node_spec: dict,
    action_module,
    judge_module,
    egonetics_client,
    emit_fn,          # async (event_type, data) → None
) -> Callable:
    """
    Generic node function factory.
    Wraps the existing executor logic inside a LangGraph-compatible async function.
    """
    node_id = node_spec["id"]
    canvas_id_from_spec = node_spec.get("canvasId", "")
    kind = node_spec.get("node_kind", "entity")
    exec_config = node_spec.get("exec_config") or {}
    if isinstance(exec_config, str):
        try:
            exec_config = json.loads(exec_config)
        except Exception:
            exec_config = {}

    async def node_fn(state: dict) -> dict:
        canvas_id = state.get("canvas_id", canvas_id_from_spec)
        task_id = state.get("task_id", "")
        ctx = {
            "task_id": task_id,
            "canvas_id": canvas_id,
            "variables": state.get("variables", {}),
            "outputs": state.get("outputs", {}),
        }

        # Update Egonetics lifecycle: running
        try:
            await egonetics_client.set_node_lifecycle(canvas_id, node_id, "running")
        except Exception:
            pass
        await emit_fn("node_start", {"node_id": node_id, "canvas_id": canvas_id, "node_kind": kind})

        traj_id = save_trajectory(task_id, canvas_id, node_id, kind, {"exec_config": exec_config})

        try:
            output, cost, error = await _dispatch(kind, exec_config, ctx, action_module, judge_module, task_id)
        except Exception as e:
            error = str(e)
            output = None
            cost = {}

        if error:
            fail_trajectory(traj_id, error)
            try:
                await egonetics_client.set_node_lifecycle(canvas_id, node_id, "failed")
            except Exception:
                pass
            await emit_fn("node_failed", {"node_id": node_id, "canvas_id": canvas_id, "error": error})
            # Append error block to exec_step page
            await _append_block(egonetics_client, node_spec, f"❌ {error}", "error")
            return {
                "trajectory": [{"node_id": node_id, "status": "failed", "error": error}],
                "failed_nodes": [node_id],
            }
        else:
            complete_trajectory(traj_id, output or {}, cost)
            try:
                await egonetics_client.set_node_lifecycle(canvas_id, node_id, "success", cost)
            except Exception:
                pass
            await emit_fn("node_complete", {"node_id": node_id, "canvas_id": canvas_id, "cost": cost})
            # Append output block to exec_step page
            out_text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)[:800]
            await _append_block(egonetics_client, node_spec, out_text, "paragraph")
            return {
                "outputs": {node_id: output},
                "variables": _extract_variables(node_id, output),
                "trajectory": [{"node_id": node_id, "status": "success", "output_preview": out_text[:200]}],
            }

    return node_fn


def make_human_gate_fn(
    node_spec: dict,
    egonetics_client,
    emit_fn,
) -> Callable:
    """Human gate node — uses LangGraph interrupt() to pause and wait for user input."""
    node_id = node_spec["id"]
    canvas_id_from_spec = node_spec.get("canvasId", "")
    exec_config = node_spec.get("exec_config") or {}
    if isinstance(exec_config, str):
        try:
            exec_config = json.loads(exec_config)
        except Exception:
            exec_config = {}

    async def node_fn(state: dict) -> dict:
        canvas_id = state.get("canvas_id", canvas_id_from_spec)
        task_id = state.get("task_id", "")
        prompt_text = exec_config.get("prompt", "需要你的确认才能继续")

        try:
            await egonetics_client.set_node_lifecycle(canvas_id, node_id, "waiting_human")
        except Exception:
            pass

        fb_id = create_feedback(
            task_id=task_id,
            feedback_type="decision_query",
            context={"config": exec_config, "node_id": node_id},
            prompt=prompt_text,
            is_blocking=True,
        )

        await emit_fn("human_gate", {
            "node_id": node_id,
            "canvas_id": canvas_id,
            "feedback_id": fb_id,
            "prompt": prompt_text,
        })

        # LangGraph interrupt: saves checkpoint, pauses, waits for Command(resume=...)
        user_response = interrupt({"feedback_id": fb_id, "prompt": prompt_text})

        try:
            await egonetics_client.set_node_lifecycle(canvas_id, node_id, "success")
        except Exception:
            pass
        await emit_fn("feedback_resolved", {"node_id": node_id, "feedback_id": fb_id})

        return {
            "outputs": {node_id: {"feedback_id": fb_id, "response": user_response}},
            "variables": {f"human_{node_id}": user_response},
            "trajectory": [{"node_id": node_id, "status": "success", "human_response": user_response}],
        }

    return node_fn


def make_condition_fn(node_spec: dict) -> Callable:
    """Returns a condition function for LangGraph conditional_edges."""
    exec_config = node_spec.get("exec_config") or {}
    if isinstance(exec_config, str):
        try:
            exec_config = json.loads(exec_config)
        except Exception:
            exec_config = {}

    node_id = node_spec["id"]
    # branches maps "true"/"false" to canvas node IDs (set at graph build time)
    branches: dict = exec_config.get("_lg_branches", {})  # populated by graph.py
    condition: str = exec_config.get("condition", "True")

    def condition_fn(state: dict) -> str:
        ctx = state.get("variables", {})
        ctx["outputs"] = state.get("outputs", {})
        try:
            result = bool(eval(condition, {"ctx": ctx, "__builtins__": {}}))
            return branches.get("true" if result else "false", "__end__")
        except Exception as e:
            logger.warning(f"rule_branch {node_id} condition eval failed: {e}")
            return "__end__"

    return condition_fn


# ── Dispatch helpers ────────────────────────────────────────────────────────

async def _dispatch(
    kind: str, cfg: dict, ctx: dict,
    action_module, judge_module, task_id: str
) -> tuple[Any, dict, str]:
    """Returns (output, cost, error_or_empty)."""
    if kind == "llm_call":
        result = await action_module.llm_call(
            prompt=cfg.get("prompt", ""),
            system=cfg.get("system", ""),
            context=ctx,
            model=cfg.get("model", settings.default_llm_model),
            max_tokens=cfg.get("budget_tokens", 4000),
        )
        return result["content"], {
            "token_input": result.get("input_tokens", 0),
            "token_output": result.get("output_tokens", 0),
        }, ""

    elif kind == "tool_call":
        result = await action_module.tool_call(cfg.get("tool", ""), cfg.get("args", {}), ctx)
        ok = result.get("ok", False)
        return result, {}, "" if ok else result.get("error", "tool_call failed")

    elif kind == "local_judge":
        judgment = await judge_module.judge(question=cfg.get("question", ""), context=ctx)
        threshold = cfg.get("confidence_threshold", settings.judge_confidence_threshold)
        if judgment.get("confidence", 0) < threshold:
            fb_id = create_feedback(
                task_id=task_id,
                feedback_type="decision_query",
                context={"question": cfg.get("question"), "judgment": judgment},
                prompt=(
                    f"模型不确定：{cfg.get('question')}\n"
                    f"模型建议：{judgment.get('answer', '?')} (置信度 {judgment.get('confidence', 0):.2f})\n"
                    "请给出你的判断："
                ),
                is_blocking=True,
            )
            return None, {}, f"low_confidence:feedback:{fb_id}"
        return judgment, {}, ""

    elif kind == "rule_branch":
        # Actual routing done by conditional edge; just mark as pass-through
        return {"branch": "evaluated"}, {}, ""

    elif kind == "lifecycle":
        from datetime import datetime
        return {"lifecycle_action": cfg.get("action", "checkpoint"),
                "timestamp": datetime.now().isoformat()}, {}, ""

    else:
        # Entity / unknown — pass through
        return {"entity_id": ctx.get("node_id", "")}, {}, ""


async def _append_block(egonetics_client, node_spec: dict, text: str, block_type: str):
    """Write output as a block to the exec_step page linked to this node."""
    try:
        entity_id = node_spec.get("entityId") or node_spec.get("entity_id")
        if entity_id:
            await egonetics_client.append_block_to_page(
                page_id=entity_id,
                block_type=block_type,
                content=text,
                creator="agent",
            )
    except Exception as e:
        logger.debug(f"append_block failed for node {node_spec.get('id')}: {e}")


def _extract_variables(node_id: str, output: Any) -> dict:
    """Extract variables from node output for downstream nodes."""
    if isinstance(output, dict):
        return {f"output_{node_id}": output}
    return {f"output_{node_id}": output}
