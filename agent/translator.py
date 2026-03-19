"""
Task Translator: writes a compiled graph definition to Egonetics.
  - Creates an execution canvas (canvas_type=execution, task_ref_id=task_id)
  - Creates exec_step pages (page_type=exec_step, ref_id=task_id)
  - Creates canvas nodes linking to those pages
  - Creates canvas relations for edges
Returns canvas_id.

Graph compilation (NL → validated graph JSON) is handled by agent/compiler.py.
"""
import json
import logging

from client.egonetics import egonetics
from agent.compiler import NL2ExecGraphCompiler

logger = logging.getLogger(__name__)


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
    Compile task NL → validated graph → write to Egonetics.
    Returns canvas_id.
    """
    # 1. Fetch task from Egonetics
    task = await egonetics.get_task(task_id)
    task_title = task.get("title") or task.get("name") or "未命名任务"
    task_desc  = task.get("taskSummary") or task.get("description") or ""

    # Enrich task_desc from page blocks
    pages = await egonetics.list_pages(root_only=False, page_type="task")
    task_page = next((p for p in pages if p.get("refId") == task_id or p.get("id") == task_id), None)
    if task_page:
        try:
            blocks = await egonetics.get_page_blocks(task_page["id"])
            block_texts = []
            for b in blocks[:20]:
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

    logger.info(f"Compiling task {task_id}: {task_title}")

    # 2. NL2ExecGraph Compiler (Steps 0-3)
    compiler = NL2ExecGraphCompiler(action_module)
    graph = await compiler.compile(task_title, task_desc)

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
