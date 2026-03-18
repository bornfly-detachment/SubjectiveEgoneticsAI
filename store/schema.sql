PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Execution trajectories: one row per node execution attempt
CREATE TABLE IF NOT EXISTS trajectories (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    canvas_id       TEXT,
    node_id         TEXT,
    node_kind       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    started_at      TEXT,
    ended_at        TEXT,
    net_time_ms     INTEGER DEFAULT 0,
    cost_vector     TEXT NOT NULL DEFAULT '{}',
    input_context   TEXT NOT NULL DEFAULT '{}',
    output_result   TEXT NOT NULL DEFAULT '{}',
    error_info      TEXT,
    reward          REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Detailed cost per execution
CREATE TABLE IF NOT EXISTS cost_records (
    id              TEXT PRIMARY KEY,
    trajectory_id   TEXT NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    time_ms         INTEGER DEFAULT 0,
    memory_mb       REAL DEFAULT 0,
    vram_mb         REAL DEFAULT 0,
    token_input     INTEGER DEFAULT 0,
    token_output    INTEGER DEFAULT 0,
    completion_rate REAL DEFAULT 0,
    quality_score   REAL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- User feedback: 4 types
CREATE TABLE IF NOT EXISTS user_feedback (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    trajectory_id   TEXT,
    feedback_type   TEXT NOT NULL,  -- 'graph_update'|'failure_analysis'|'decision_query'|'value_judgment'
    context         TEXT NOT NULL DEFAULT '{}',
    prompt          TEXT,           -- what was asked to user
    user_response   TEXT,
    resolved        INTEGER NOT NULL DEFAULT 0,
    is_blocking     INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at     TEXT
);

-- Failure cases: saved for analysis + training
CREATE TABLE IF NOT EXISTS failure_cases (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    node_id         TEXT,
    failure_type    TEXT NOT NULL,  -- 'timeout'|'loop'|'budget'|'capability'|'judgment_error'
    trajectory_json TEXT NOT NULL DEFAULT '[]',
    root_cause      TEXT,
    solution        TEXT,
    training_sample TEXT,           -- generated SFT/GRPO sample
    analyzed        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Model versions
CREATE TABLE IF NOT EXISTS model_versions (
    id              TEXT PRIMARY KEY,
    version_num     INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,
    training_type   TEXT NOT NULL,  -- 'sft'|'grpo'
    base_version    TEXT,           -- parent version id
    training_samples INTEGER DEFAULT 0,
    reward_avg      REAL,
    eval_loss       REAL,
    is_active       INTEGER NOT NULL DEFAULT 0,
    ab_test_result  TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Constitution translation: human language → machine rules
CREATE TABLE IF NOT EXISTS constitution_rules (
    id              TEXT PRIMARY KEY,
    source_text     TEXT NOT NULL,  -- original human language principle
    rule_type       TEXT NOT NULL,  -- 'value_filter'|'priority_weight'|'direction_gate'
    machine_rule    TEXT,           -- extracted if/then rule
    confidence      REAL DEFAULT 0,
    trigger_count   INTEGER DEFAULT 0,
    success_count   INTEGER DEFAULT 0,
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_traj_task ON trajectories(task_id);
CREATE INDEX IF NOT EXISTS idx_traj_status ON trajectories(status);
CREATE INDEX IF NOT EXISTS idx_feedback_task ON user_feedback(task_id);
CREATE INDEX IF NOT EXISTS idx_feedback_resolved ON user_feedback(resolved);
CREATE INDEX IF NOT EXISTS idx_failure_task ON failure_cases(task_id);
CREATE INDEX IF NOT EXISTS idx_model_active ON model_versions(is_active);
