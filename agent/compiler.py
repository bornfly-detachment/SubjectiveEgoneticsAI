"""
NL2ExecGraph Compiler
=====================
4-step pipeline: Human NL → validated, runnable LangGraph-ready graph definition.

Step 0 — Subjectivity Interpreter
    Calls local model to check meta-control principles + interpret user intent.
    Knows who the user IS before decomposing anything.

Step 1+2 — Task Decomposer + Capability Mapper
    Calls LLM API with enriched context (intent + capability boundary rules).
    Produces structured graph JSON.

Step 3 — Graph Validator + Self-correction
    Validates LangGraph rules. Loops back with errors if invalid (max 3 retries).
"""
import json
import logging
import httpx
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Capability boundary table (injected into LLM system prompt) ──────────────

CAPABILITY_RULES = """\
【能力边界映射规则】
根据子任务特征，映射到对应 node_kind：

| 子任务特征                          | node_kind    |
|------------------------------------|--------------|
| 推理、分析、写作、规划、总结          | llm_call     |
| 执行代码、修改文件、读写文件系统       | tool_call (tool: claude_code)  |
| 需要外部 session、持续对话、web 搜索  | tool_call (tool: openclaw)     |
| 价值判断、路径选择、不确定性路由       | local_judge  |
| 根据变量/输出结果条件分支             | rule_branch  |
| 需要人类决策、涉及伦理/权限边界       | human_gate   |
| 生命周期控制（开始/结束标记）         | lifecycle    |

【LangGraph 合法图规则（必须满足，否则无法编译）】
1. 必须有且仅有一个入口节点（无入边的节点）
2. 必须有至少一个叶子节点（无出边，将映射到 END）
3. edges 中引用的所有 index 必须在 nodes 中存在
4. rule_branch 节点的 exec_config 必须包含 branches: {"true": <index>, "false": <index>}
5. rule_branch 的 branch 目标 index 必须存在于 nodes 中
6. 所有节点必须可从入口节点到达（无孤岛节点）
7. 节点数量：简单任务 3-6 个，复杂任务 7-12 个，不要过度设计
"""

# ── Step 1+2 System Prompt ─────────────────────────────────────────────────

COMPILE_SYSTEM = """\
你是 NL2ExecGraph Compiler，将用户任务翻译为机器控制论执行图。

{capability_rules}

【node_kind 的 exec_config 格式】
- lifecycle:    {{"action": "start" | "complete"}}
- llm_call:     {{"prompt": "...", "system": "...", "model": "{default_model}", "budget_tokens": 4000}}
- tool_call:    {{"tool": "claude_code" | "openclaw", "args": {{...}}}}
- local_judge:  {{"question": "...", "confidence_threshold": 0.6}}
- rule_branch:  {{"condition": "ctx['变量名'] == '值'", "branches": {{"true": <index>, "false": <index>}}}}
- human_gate:   {{"prompt": "需要确认的问题", "blocking": true}}

【输出格式（严格 JSON，不包含任何其他内容）】
{{
  "title": "执行图标题",
  "nodes": [
    {{"index": 0, "title": "节点名", "icon": "emoji", "node_kind": "lifecycle", "exec_config": {{"action": "start"}}}},
    ...
  ],
  "edges": [
    {{"from": 0, "to": 1}},
    ...
  ]
}}

第一个节点必须是 lifecycle(start)，最后一个必须是 lifecycle(complete)。
"""

# ── Validation ─────────────────────────────────────────────────────────────

def validate_graph(graph: dict) -> list[str]:
    """Returns list of validation errors. Empty list = valid."""
    errors = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not nodes:
        errors.append("nodes 列表为空")
        return errors

    indices = {n["index"] for n in nodes}

    # Check edge references
    for e in edges:
        if e["from"] not in indices:
            errors.append(f"edge.from={e['from']} 不存在于 nodes")
        if e["to"] not in indices:
            errors.append(f"edge.to={e['to']} 不存在于 nodes")

    # Check entry point (node with no incoming edges)
    has_incoming = {e["to"] for e in edges}
    entry_points = [n for n in nodes if n["index"] not in has_incoming]
    if not entry_points:
        errors.append("没有入口节点（所有节点都有入边，存在环路）")
    elif len(entry_points) > 1:
        errors.append(f"有多个入口节点: {[n['index'] for n in entry_points]}，应只有一个")

    # Check rule_branch nodes
    for n in nodes:
        if n.get("node_kind") == "rule_branch":
            cfg = n.get("exec_config", {})
            branches = cfg.get("branches", {})
            if "true" not in branches or "false" not in branches:
                errors.append(f"rule_branch 节点 {n['index']} 缺少 branches.true 或 branches.false")
            for k, target in branches.items():
                if target not in indices:
                    errors.append(f"rule_branch 节点 {n['index']} branch.{k}={target} 不存在")

    # Check reachability (simple BFS from entry)
    if entry_points:
        adj: dict[int, list[int]] = {}
        for e in edges:
            adj.setdefault(e["from"], []).append(e["to"])
        visited = set()
        queue = [entry_points[0]["index"]]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            queue.extend(adj.get(cur, []))
        unreachable = indices - visited
        if unreachable:
            errors.append(f"孤岛节点（不可达）: {unreachable}")

    return errors


@dataclass
class InterpretedIntent:
    """Output of Step 0: subjectivity-filtered task intent."""
    task_title: str
    task_desc: str
    intent_summary: str          # What does this task REALLY mean for this user
    value_annotations: list[str] = field(default_factory=list)  # e.g. ["需要human_gate确认", "优先本地模型判断"]
    blocked: bool = False        # True = task conflicts with meta-control principles
    block_reason: str = ""


# ── Compiler ────────────────────────────────────────────────────────────────

class NL2ExecGraphCompiler:

    def __init__(self, action_module):
        self.action = action_module

    # ── Step 0: Subjectivity Interpreter ──────────────────────────────────

    async def _subjectivity_check(self, task_title: str, task_desc: str) -> InterpretedIntent:
        """
        Ask local model: does this task align with the user's meta-control principles?
        Falls back to passthrough if local model is unavailable.
        """
        base = InterpretedIntent(
            task_title=task_title,
            task_desc=task_desc,
            intent_summary=f"{task_title}. {task_desc[:200]}".strip(),
        )

        question = (
            f"任务标题：{task_title}\n"
            f"任务描述：{task_desc[:500] or '（无）'}\n\n"
            "请判断：\n"
            "1. 此任务是否符合主体性原则？（是否触碰不可突破的元控制边界？）\n"
            "2. 在用户的主体性语境下，此任务真正意图是什么？\n"
            "3. 执行时有哪些价值判断注意事项？（如：需人工确认、优先本地判断、谨慎操作）\n"
            "输出 JSON: {\"aligned\": true/false, \"block_reason\": \"...\", "
            "\"intent_summary\": \"...\", \"value_annotations\": [\"...\"]}"
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{settings.inference_url}/judge",
                    json={"question": question, "context": {}},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Local model returns {"answer", "confidence", "reasoning"}
                    # Try to parse structured intent from reasoning
                    try:
                        reasoning = data.get("reasoning", "")
                        # Look for JSON in reasoning
                        if "{" in reasoning:
                            raw = reasoning[reasoning.index("{"):reasoning.rindex("}") + 1]
                            parsed = json.loads(raw)
                            if not parsed.get("aligned", True):
                                base.blocked = True
                                base.block_reason = parsed.get("block_reason", "本地模型判断不符合主体性原则")
                            base.intent_summary = parsed.get("intent_summary", base.intent_summary)
                            base.value_annotations = parsed.get("value_annotations", [])
                    except Exception:
                        # Unstructured output — use confidence as alignment signal
                        if data.get("answer", "").startswith("否") and data.get("confidence", 1.0) > 0.8:
                            base.blocked = True
                            base.block_reason = data.get("reasoning", "")
        except Exception as e:
            logger.debug(f"Local model unavailable for subjectivity check: {e}")
            # Passthrough — local model not running yet, skip Step 0

        return base

    # ── Step 1+2: Decompose + Map ──────────────────────────────────────────

    async def _decompose_and_map(self, intent: InterpretedIntent) -> dict:
        """Call LLM with enriched context → raw graph JSON."""
        system = COMPILE_SYSTEM.format(
            capability_rules=CAPABILITY_RULES,
            default_model=settings.default_llm_model,
        )

        # Enrich prompt with intent + value annotations
        annotations_str = ""
        if intent.value_annotations:
            annotations_str = "\n【价值标注（来自主体性解读）】\n" + "\n".join(
                f"- {a}" for a in intent.value_annotations
            )

        prompt = (
            f"任务标题：{intent.task_title}\n\n"
            f"主体性意图解读：{intent.intent_summary}\n"
            f"{annotations_str}\n\n"
            f"原始描述：\n{intent.task_desc or '（无）'}"
        )

        result = await self.action.llm_call(
            prompt=prompt,
            system=system,
            model=settings.default_llm_model,
            max_tokens=2500,
        )
        raw = result["content"].strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        # Handle trailing fences
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")].strip()

        return json.loads(raw)

    # ── Step 3: Validate + Self-correct ───────────────────────────────────

    async def _validate_and_fix(self, graph: dict, intent: InterpretedIntent,
                                 max_retries: int = 3) -> dict:
        """Validate graph and self-correct via LLM if invalid."""
        for attempt in range(max_retries):
            errors = validate_graph(graph)
            if not errors:
                return graph

            logger.warning(f"Graph validation failed (attempt {attempt + 1}): {errors}")

            fix_prompt = (
                f"以下执行图定义存在错误，请修复后重新输出合法的 JSON：\n\n"
                f"当前图定义：\n{json.dumps(graph, ensure_ascii=False, indent=2)}\n\n"
                f"错误列表：\n" + "\n".join(f"- {e}" for e in errors) +
                f"\n\n任务意图：{intent.intent_summary}\n\n"
                f"只输出修复后的 JSON，不要其他内容。"
            )

            result = await self.action.llm_call(
                prompt=fix_prompt,
                system=COMPILE_SYSTEM.format(
                    capability_rules=CAPABILITY_RULES,
                    default_model=settings.default_llm_model,
                ),
                model=settings.default_llm_model,
                max_tokens=2500,
            )
            raw = result["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")].strip()

            try:
                graph = json.loads(raw)
            except Exception as e:
                logger.error(f"Fix attempt {attempt + 1} JSON parse failed: {e}")
                continue

        # All retries exhausted — return minimal valid fallback
        logger.error("All validation retries exhausted, using minimal fallback graph")
        return {
            "title": intent.task_title,
            "nodes": [
                {"index": 0, "title": "开始", "icon": "🚀", "node_kind": "lifecycle",
                 "exec_config": {"action": "start"}},
                {"index": 1, "title": intent.task_title, "icon": "🤖", "node_kind": "llm_call",
                 "exec_config": {"prompt": intent.intent_summary, "budget_tokens": 4000}},
                {"index": 2, "title": "完成", "icon": "✅", "node_kind": "lifecycle",
                 "exec_config": {"action": "complete"}},
            ],
            "edges": [{"from": 0, "to": 1}, {"from": 1, "to": 2}],
        }

    # ── Public entry ──────────────────────────────────────────────────────

    async def compile(self, task_title: str, task_desc: str) -> dict:
        """
        Full 4-step compilation.
        Returns validated graph dict ready for Egonetics + LangGraph.
        Raises RuntimeError if task is blocked by meta-control.
        """
        # Step 0: Subjectivity check
        intent = await self._subjectivity_check(task_title, task_desc)
        logger.info(f"[Compiler] Step0 done: blocked={intent.blocked} intent={intent.intent_summary[:80]}")

        if intent.blocked:
            raise RuntimeError(f"任务被元控制层拦截: {intent.block_reason}")

        # Step 1+2: Decompose + Capability Map
        try:
            graph = await self._decompose_and_map(intent)
        except Exception as e:
            logger.error(f"[Compiler] Step1+2 failed: {e}")
            raise

        logger.info(f"[Compiler] Step1+2 done: {len(graph.get('nodes', []))} nodes")

        # Step 3: Validate + self-correct
        graph = await self._validate_and_fix(graph, intent)
        logger.info(f"[Compiler] Step3 done: graph valid, {len(graph.get('nodes', []))} nodes")

        return graph
