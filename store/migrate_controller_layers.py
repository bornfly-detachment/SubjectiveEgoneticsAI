"""
迁移脚本：为 execution_queue 添加三层控制器字段
运行: python store/migrate_controller_layers.py
"""
from store.db import get_conn

COLUMNS = [
    ("layer",         "TEXT    NOT NULL DEFAULT 'local'"),  # 'global'|'conflict'|'local'
    ("dependencies",  "TEXT    NOT NULL DEFAULT '[]'"),     # JSON 依赖节点 ID 列表（全局层用）
    ("regen_count",   "INTEGER NOT NULL DEFAULT 0"),        # 已重新生成次数（全局层）
    ("max_regen",     "INTEGER NOT NULL DEFAULT 3"),        # 最大重新生成次数
    ("conflict_type", "TEXT"),                              # 冲突类型（冲突层）
    ("options",       "TEXT    NOT NULL DEFAULT '[]'"),     # JSON 选项列表（冲突层）
    ("global_ref_id", "TEXT"),                              # 所属全局节点 ID（局部层）
    ("resolved",      "INTEGER NOT NULL DEFAULT 0"),        # 冲突是否已解决
]

def migrate():
    conn = get_conn()
    cur = conn.execute("PRAGMA table_info(execution_queue)")
    existing = {row["name"] for row in cur.fetchall()}

    added = 0
    for col_name, col_def in COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE execution_queue ADD COLUMN {col_name} {col_def}")
            print(f"  + added column: {col_name}")
            added += 1
        else:
            print(f"  · skip (exists): {col_name}")

    conn.commit()
    conn.close()
    print(f"\nMigration done. {added} column(s) added.")

if __name__ == "__main__":
    migrate()
