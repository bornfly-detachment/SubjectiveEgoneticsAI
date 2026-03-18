"""Model version management."""
import logging
from store.db import get_conn, row_to_dict, gen_id
logger = logging.getLogger(__name__)

def register_version(checkpoint_path, training_type, training_samples, reward_avg=None, eval_loss=None, base_version=None):
    conn = get_conn()
    last = conn.execute("SELECT MAX(version_num) as v FROM model_versions").fetchone()
    version_num = (last["v"] or 0) + 1
    vid = gen_id("ver")
    conn.execute("INSERT INTO model_versions (id,version_num,checkpoint_path,training_type,base_version,training_samples,reward_avg,eval_loss,is_active) VALUES (?,?,?,?,?,?,?,?,0)",
                 [vid, version_num, checkpoint_path, training_type, base_version, training_samples, reward_avg, eval_loss])
    conn.commit(); conn.close()
    return vid

def activate_version(version_id):
    conn = get_conn()
    conn.execute("UPDATE model_versions SET is_active=0")
    conn.execute("UPDATE model_versions SET is_active=1 WHERE id=?", [version_id])
    conn.commit()
    ver = row_to_dict(conn.execute("SELECT * FROM model_versions WHERE id=?", [version_id]).fetchone())
    conn.close()
    return ver

def get_active_version():
    conn = get_conn()
    row = conn.execute("SELECT * FROM model_versions WHERE is_active=1").fetchone()
    conn.close()
    return row_to_dict(row) if row else None

def list_versions():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM model_versions ORDER BY version_num DESC").fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]
