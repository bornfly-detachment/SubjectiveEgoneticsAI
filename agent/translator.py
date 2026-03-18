"""
Task Translator: human language Task → execution Graph
Calls LLM to produce a structured execution plan, then writes it to Egonetics:
  - Creates an execution canvas (canvas_type=execution, task_ref_id=task_id)
  - Creates exec_step pages (page_type=exec_step, ref_id=task_id)
  - Creates canvas nodes linking to those pages
Returns canvas_id.
"""
import json
import logging
from datetime import datetime

from client.egonetics import egonetics
from config.settings import settings

logger = logging.getLogger(__name__)

TRANSLATE_SYSTEM = """\
你是一个执行图生成器。将用户任务转换为机器控制论执行图，输出严格的 JSON。

node_kind 类型说明：
- lifecycle   生命周期控制节点，exec_config: {"action": "start"|"checkpoint"|"complete"}
- llm_call    调用大模型推理，exec_config: {"prompt": "...", "system": "...", "model": "ark-code-latest", "budget_tokens": 4000}
- tool_call   调用工具，exec_config: {"tool": "claude_code"|"openclaw"|"read_file"|"write_file", "args": {...}}
- local_judge 本地模型价值判断，exec_config: {"question": "...", "confidence_threshold": 0.6}
- rule_branch 规则分支，exec_config: {"condition": "ctx['变量名'] == '值'", "branches": {"true": <节点序号>, "false": <节点序号>}}
- human_gate  人工介入，exec_config: {"prompt": "需要确认的问题", "blocking": true}

输出格式（只输出 JSON，不要任何其他内容）：
{
  "title": "执行图标题",
  "nodes": [
    {
      "index": 0,
      "title": "节点名称",
      "icon": "emoji",
      "node_kind": "lifecycle",
      "exec_config": {"action": "start"}
    }
  ],
  "edges": [
    {"from": 0, "to": 1}
  ]
}

规则：
1. 第一个节点必须是 lifecycle(start)，最后一个必须是 lifecycle(complete)
2. 编码/文件操作任务用 tool_call:claude_code
3. 多步协作任务或需要 session 的用 tool_call:openclaw
4. 分析/推理用 llm_call
5. 涉及价值判断/路由决策用 local_judge
6. 边界不确定、需要用户确认用 human_gate
7. 节点数量：简单任务 3-5 个，复杂任务 6-10 个，不要过度设计
"""


def _layout_positions(nodes: list, edges: list) -> list:
    """Assign x/y positions based on topological order."""
    # Build adjacency for topo sort
    in_degree = {n["index"]: 0 for n in nodes}
    children = {n["index"]: [] for n in nodes}
    for e in edges:
        in_degree[e["to"]] = in_degree.get(e["to"], 0) + 1
        children[e["from"]].append(e["to"])

    # BFS layers
    queue = [i for i, deg in in_degree.items() if deg == 0]
    layers = []
    visited = set()
    while queue:
        layers.append(queue[:])
        next_q = []
        for n in queue:
            visited.add(n)
            for c in children.get(n, []):
                in_degree[c] -= 1
                if in_degree[c] == 0 and c not in visited:
                    next_q.append(c)
        queue = next_q

    pos = {}
    x_start, y_start = 120, 120
    x_gap, y_gap = 280, 140
    for layer_idx, layer in enumerate(layers):
        for row_idx, node_idx in enumerate(layer):
            # Center rows vertically
            y_offset = (row_idx - (len(layer) - 1) / 2) * y_gap
            pos[node_idx] = {
                "x": x_start + layer_idx * x_gap,
                "y": y_start + y_offset,
            }

    # Fallback for any missed nodes
    for n in nodes:
        if n["index"] not in pos:
            pos[n["index"]] = {"x": x_start, "y": y_start + n["index"] * y_gap}

    return pos


async def translate(task_id: str, action_module) -> str:
    """
    Translate a Task into an execution Canvas.
    Returns canvas_id.
    """
    # 1. Fetch task from Egonetics
    task = await egonetics.get_task(task_id)
    task_title = task.get("title") or task.get("name") or "未命名任务"
    task_desc  = task.get("taskSummary") or task.get("description") or ""

    # Fetch task page blocks for richer context
    pages = await egonetics.list_pages(root_only=False, page_type="task")
    task_page = next((p for p in pages if p.get("refId") == task_id or p.get("id") == task_id), None)
    if task_page:
        try:
            blocks = await egonetics.get_page_blocks(task_page["id"])
            # Extract text content from blocks
            block_texts = []
            for b in blocks[:20]:  # limit context
                content = b.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        pass
                text = content.get("text", "") or content.get("value", "")
                if text:
                    block_texts.append(text)
            if block_texts:
                task_desc = "\n".join(block_texts)
        except Exception as e:
            logger.warning(f"Cannot fetch task blocks: {e}")

    user_prompt = f"任务标题：{task_title}\n\n任务描述：\n{task_desc or '（无描述）'}"
    logger.info(f"Translating task {task_id}: {task_title}")

    # 2. Call LLM to generate execution graph
    result = await action_module.llm_call(
        prompt=user_prompt,
        system=TRANSLATE_SYSTEM,
        model=settings.default_llm_model,
        max_tokens=2000,
    )
    raw = result["content"].strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        graph = json.loads(raw)
    except Exception as e:
        logger.error(f"Graph JSON parse failed: {e}\nRaw: {raw[:300]}")
        # Fallback: minimal 3-node graph
        graph = {
            "title": task_title,
            "nodes": [
                {"index": 0, "title": "开始", "icon": "🚀", "node_kind": "lifecycle",
                 "exec_config": {"action": "start"}},
                {"index": 1, "title": task_title, "icon": "🤖", "node_kind": "llm_call",
                 "exec_config": {"prompt": user_prompt, "budget_tokens": 4000}},
                {"index": 2, "title": "完成", "icon": "✅", "node_kind": "lifecycle",
                 "exec_config": {"action": "complete"}},
            ],
            "edges": [{"from": 0, "to": 1}, {"from": 1, "to": 2}],
        }

    nodes  = graph.get("nodes", [])
    edges  = graph.get("edges", [])
    canvas_title = graph.get("title", task_title)
    positions = _layout_positions(nodes, edges)

    # 3. Create execution canvas in Egonetics
    canvas = await egonetics.create_execution_canvas(
        task_id=task_id,
        title=f"[执行] {canvas_title}",
    )
    canvas_id = canvas["id"]
    logger.info(f"Created execution canvas: {canvas_id}")

    # 4. Create exec_step pages + canvas nodes
    index_to_node_id = {}
    for node_spec in nodes:
        idx   = node_spec["index"]
        title = node_spec.get("title", f"节点{idx}")
        icon  = node_spec.get("icon", "⚙️")
        kind  = node_spec.get("node_kind", "llm_call")
        cfg   = node_spec.get("exec_config", {})

        # Create exec_step page (independent, linked to task via ref_id)
        page = await egonetics.create_exec_step_page(
            task_id=task_id,
            title=title,
            icon=icon,
        )
        page_id = page["id"]

        # Create canvas node linking to that page
        node = await egonetics.add_node(
            canvas_id=canvas_id,
            entity_type="exec_step",
            entity_id=page_id,
            x=positions[idx]["x"],
            y=positions[idx]["y"],
            node_kind=kind,
            exec_config=cfg,
        )
        index_to_node_id[idx] = node["id"]

    # 5. Create relations (edges) between canvas nodes as page relations
    for edge in edges:
        src_node_id = index_to_node_id.get(edge["from"])
        tgt_node_id = index_to_node_id.get(edge["to"])
        if src_node_id and tgt_node_id:
            try:
                await egonetics.create_relation(
                    source_type="canvas_node",
                    source_id=src_node_id,
                    target_type="canvas_node",
                    target_id=tgt_node_id,
                    relation_type="chain",
                    title="",
                )
            except Exception as e:
                logger.warning(f"Edge {edge['from']}→{edge['to']} relation failed: {e}")

    logger.info(f"Translation complete: canvas={canvas_id}, nodes={len(nodes)}, edges={len(edges)}")
    return canvas_id
