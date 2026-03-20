"""
LangGraph StateGraph builder.
Reads Egonetics canvas nodes + relations → builds a compiled StateGraph
with SQLite checkpointing.
"""
import logging
from typing import Annotated, Any
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from client.egonetics import egonetics
from agent.nodes import make_node_fn, make_human_gate_fn, make_condition_fn
from config.settings import settings

logger = logging.getLogger(__name__)


# ── State Schema ─────────────────────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class ExecState(TypedDict):
    task_id: str
    canvas_id: str
    outputs:  Annotated[dict, _merge_dicts]       # node_id → output (merged on update)
    variables: Annotated[dict, _merge_dicts]       # runtime variables
    trajectory: Annotated[list, operator.add]      # append-only execution log
    failed_nodes: Annotated[list, operator.add]    # append-only failed node list


# ── Graph Builder ─────────────────────────────────────────────────────────────

async def build_graph(
    canvas_id: str,
    action_module,
    judge_module,
    emit_fn,
    checkpointer=None,
):
    """
    Fetch canvas nodes + relations from Egonetics,
    build a LangGraph StateGraph, and compile it.

    Returns (compiled_graph, checkpointer).
    The checkpointer is owned by the caller if passed in; otherwise a new
    SqliteSaver is created and returned for the caller to close.
    """
    nodes = await egonetics.get_nodes(canvas_id)

    if not nodes:
        raise ValueError(f"Canvas {canvas_id} has no nodes")

    # Fetch relations per canvas node (relations are stored with node_id as source_id)
    relations: list[dict] = []
    for node in nodes:
        node_rels = await egonetics.get_relations(source_id=node["id"])
        relations.extend(node_rels)

    # Build adjacency: node_id → [child node_ids]
    adj: dict[str, list[str]] = {}
    for rel in relations:
        src = rel.get("source_id") or rel.get("sourceId")
        tgt = rel.get("target_id") or rel.get("targetId")
        if src and tgt:
            adj.setdefault(src, []).append(tgt)

    has_incoming = {
        tgt
        for rel in relations
        for tgt in [rel.get("target_id") or rel.get("targetId")]
        if tgt
    }
    node_map = {n["id"]: n for n in nodes}

    # ── Build StateGraph ──
    builder = StateGraph(ExecState)

    for node in nodes:
        nid = node["id"]
        kind = node.get("node_kind", "entity")

        if kind == "human_gate":
            fn = make_human_gate_fn(node, egonetics, emit_fn)
        else:
            fn = make_node_fn(node, action_module, judge_module, egonetics, emit_fn)

        builder.add_node(nid, fn)

    # ── Add edges ──
    for node in nodes:
        nid = node["id"]
        children = adj.get(nid, [])
        kind = node.get("node_kind", "entity")

        if nid not in has_incoming:
            builder.add_edge(START, nid)

        if kind == "rule_branch":
            # Inject branch targets as LangGraph node IDs into exec_config
            exec_config = node.get("exec_config") or {}
            import json as _json
            if isinstance(exec_config, str):
                try:
                    exec_config = _json.loads(exec_config)
                except Exception:
                    exec_config = {}
            raw_branches = exec_config.get("branches", {})
            # raw_branches: {"true": <index>, "false": <index>}
            # We need canvas node IDs, not indices.
            # During compile(), nodes were created with index_to_node_id mapping.
            # Here we need to look up by… actually we need to store the index→node_id map.
            # Since we can't recover index from here easily, store _lg_branches by node title lookup.
            # Workaround: inject _lg_branches as actual node IDs when possible.
            # If branches reference are actual node IDs → use directly.
            lg_branches = {}
            for branch_key, target in raw_branches.items():
                if isinstance(target, str) and target in node_map:
                    lg_branches[branch_key] = target
                else:
                    # Try to find a node at that position in children list
                    if children:
                        lg_branches[branch_key] = children[0] if branch_key == "true" else (
                            children[1] if len(children) > 1 else children[0]
                        )
            exec_config["_lg_branches"] = lg_branches
            node["exec_config"] = exec_config

            condition_fn = make_condition_fn(node)
            # Add conditional edges to all possible targets + __end__
            possible_targets = list(set(lg_branches.values())) + [END]
            builder.add_conditional_edges(nid, condition_fn)

        elif not children:
            builder.add_edge(nid, END)
        else:
            for child in children:
                builder.add_edge(nid, child)

    # ── Compile with checkpointer ──
    owns_checkpointer = checkpointer is None
    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = builder.compile(checkpointer=checkpointer)
    logger.info(f"Built LangGraph: canvas={canvas_id}, nodes={len(nodes)}, edges={len(relations)}")

    return compiled, checkpointer, owns_checkpointer
