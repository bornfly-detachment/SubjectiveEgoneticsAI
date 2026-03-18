"""Training data generator: trajectories + feedback → SFT/GRPO samples."""
import json, logging
from pathlib import Path
from store.db import get_conn, row_to_dict
from config.settings import settings

logger = logging.getLogger(__name__)

def generate_sft_from_feedback(output_path: str = None) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM user_feedback WHERE feedback_type='decision_query' AND resolved=1 AND user_response IS NOT NULL ORDER BY resolved_at DESC"
    ).fetchall()
    conn.close()
    samples = []
    for r in rows:
        fb = row_to_dict(r)
        ctx = fb.get("context", {})
        question = ctx.get("question", fb.get("prompt", ""))
        answer = fb.get("user_response", "")
        if question and answer:
            samples.append({"instruction": question, "input": json.dumps(ctx.get("model_judgment",{}), ensure_ascii=False), "output": answer})
    if output_path and samples:
        Path(output_path).write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    return samples

def generate_grpo_from_trajectories(output_path: str = None) -> list:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM trajectories WHERE status IN ('success','failed') ORDER BY created_at DESC LIMIT 500").fetchall()
    conn.close()
    samples = []
    for r in rows:
        t = row_to_dict(r)
        cost = t.get("cost_vector", {})
        reward = _compute_reward(t["status"], cost)
        samples.append({"prompt": json.dumps({"node_kind": t["node_kind"], "input": t.get("input_context",{})}, ensure_ascii=False),
                         "response": json.dumps(t.get("output_result",{}), ensure_ascii=False), "reward": reward})
    if output_path and samples:
        Path(output_path).write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    return samples

def _compute_reward(status: str, cost: dict) -> float:
    if status == "failed": return -1.0
    time_p = min(cost.get("time_ms", 0) / 60000, 0.5)
    token_p = min((cost.get("token_input",0) + cost.get("token_output",0)) / 10000, 0.3)
    return round(1.0 - time_p - token_p, 3)

def should_trigger_sft() -> bool:
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM user_feedback WHERE feedback_type='decision_query' AND resolved=1").fetchone()[0]
    conn.close()
    return count >= settings.sft_trigger_count

def should_trigger_grpo() -> bool:
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM trajectories WHERE status IN ('success','failed')").fetchone()[0]
    conn.close()
    return count >= settings.grpo_trigger_count
