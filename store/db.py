import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from config.settings import settings

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    schema = Path(__file__).parent / "schema.sql"
    conn = get_conn()
    conn.executescript(schema.read_text())
    conn.commit()
    conn.close()

def gen_id(prefix: str = "rec") -> str:
    return f"{prefix}-{int(datetime.now().timestamp()*1000)}-{uuid.uuid4().hex[:6]}"

def row_to_dict(row) -> dict:
    if row is None:
        return None
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, str) and v.startswith(('{', '[')):
            try:
                d[k] = json.loads(v)
            except Exception:
                pass
    return d

# --- Trajectory helpers ---

def save_trajectory(task_id: str, canvas_id: str, node_id: str, node_kind: str, input_ctx: dict) -> str:
    tid = gen_id("traj")
    conn = get_conn()
    conn.execute(
        """INSERT INTO trajectories (id, task_id, canvas_id, node_id, node_kind, status, started_at, input_context)
           VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
        [tid, task_id, canvas_id, node_id, node_kind, datetime.now().isoformat(), json.dumps(input_ctx)]
    )
    conn.commit(); conn.close()
    return tid

def complete_trajectory(tid: str, result: dict, cost: dict, reward: float = None):
    conn = get_conn()
    now = datetime.now().isoformat()
    conn.execute(
        """UPDATE trajectories SET status='success', ended_at=?, output_result=?, cost_vector=?, reward=? WHERE id=?""",
        [now, json.dumps(result), json.dumps(cost), reward, tid]
    )
    conn.commit(); conn.close()

def fail_trajectory(tid: str, error: str, failure_type: str = "capability"):
    conn = get_conn()
    now = datetime.now().isoformat()
    conn.execute(
        "UPDATE trajectories SET status='failed', ended_at=?, error_info=? WHERE id=?",
        [now, error, tid]
    )
    conn.commit(); conn.close()
    # Save failure case
    save_failure_case(tid, failure_type, error)

def save_failure_case(trajectory_id: str, failure_type: str, error_info: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM trajectories WHERE id=?", [trajectory_id]).fetchone()
    if not row:
        conn.close(); return
    fid = gen_id("fail")
    conn.execute(
        """INSERT INTO failure_cases (id, task_id, node_id, failure_type, trajectory_json)
           VALUES (?, ?, ?, ?, ?)""",
        [fid, row["task_id"], row["node_id"], failure_type, json.dumps(row_to_dict(row))]
    )
    conn.commit(); conn.close()

# --- User feedback helpers ---

def create_feedback(task_id: str, feedback_type: str, context: dict, prompt: str, is_blocking: bool = True) -> str:
    fid = gen_id("fb")
    conn = get_conn()
    conn.execute(
        """INSERT INTO user_feedback (id, task_id, feedback_type, context, prompt, is_blocking)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [fid, task_id, feedback_type, json.dumps(context), prompt, 1 if is_blocking else 0]
    )
    conn.commit(); conn.close()
    return fid

def resolve_feedback(feedback_id: str, user_response: str):
    conn = get_conn()
    conn.execute(
        "UPDATE user_feedback SET resolved=1, user_response=?, resolved_at=? WHERE id=?",
        [user_response, datetime.now().isoformat(), feedback_id]
    )
    conn.commit(); conn.close()

def get_pending_feedback(task_id: str) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM user_feedback WHERE task_id=? AND resolved=0 ORDER BY created_at",
        [task_id]
    ).fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]
