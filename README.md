# SubjectiveEgoneticsAI

**Self-cybernetics autonomous agent execution engine.**

SubjectiveEgoneticsAI is the AI brain that drives the [Egonetics](https://github.com/bornfly-detachment/egonetics) personal agent system. It receives tasks from Egonetics, translates them into execution graphs, orchestrates multi-agent workflows (Claude Code + OpenClaw + local LLM), and writes results back — all observable in real time via WebSocket.

---

## Architecture

```
Egonetics (frontend + backend)
        │
        │  REST API (task/canvas/page/block)
        ▼
SubjectiveEgoneticsAI  ─────────────────────────────────────
│                                                           │
│  api/          FastAPI on :8000                           │
│   └─ lifecycle  start / stop / status / feedback / ws     │
│                                                           │
│  agent/        Execution engine                           │
│   ├─ translator  Task description → Graph JSON (LLM)      │
│   ├─ executor    Node-level dispatch (llm_call/tool_call)  │
│   ├─ loop        DFS traversal + backtracking             │
│   └─ backtrack   Failure analysis & retry strategy        │
│                                                           │
│  modules/      Capabilities                               │
│   ├─ action     LLM API calls (Anthropic / OpenAI compat) │
│   ├─ judge      Confidence scoring                        │
│   └─ agents/   Sub-agent wrappers                         │
│       ├─ claude_code   claude -p "..." --output-format json│
│       └─ openclaw      openclaw agent --message "..."     │
│                                                           │
│  models/       Local model (Qwen2-0.5B edge inference)    │
│   ├─ inference   vLLM-style FastAPI on :8001              │
│   └─ training/   SFT + GRPO via LLaMA-Factory             │
│                                                           │
│  client/       Egonetics REST client                      │
│  store/        SQLite (trajectories, feedback)            │
│  data/         SFT training data (git-ignored)            │
└───────────────────────────────────────────────────────────
```

### Multi-agent flow

1. User selects a task in Egonetics `/agents` and clicks **启动生命周期**
2. `POST /lifecycle/start` → TaskTranslator calls LLM → produces Graph JSON
3. Execution canvas + exec_step pages are created in Egonetics
4. `AgentLoop` DFS-traverses the graph, calling `NodeExecutor` per node
5. Each node may dispatch to: `llm_call`, `claude_code`, `openclaw`, `read_file`, etc.
6. Lifecycle state (`pending → running → success/failed`) is synced back to Egonetics via REST
7. Real-time events broadcast over WebSocket `/lifecycle/ws/{task_id}`
8. Human-gate nodes emit `waiting_human` and pause until `/lifecycle/feedback/{id}` is resolved

---

## Quick Start

### Prerequisites

- Python ≥ 3.11
- [Egonetics](https://github.com/bornfly-detachment/egonetics) backend running on `:3002`
- An Egonetics **agent** role JWT token (30-day expiry)
- An LLM API key (Anthropic / ARK / any Anthropic-compatible endpoint)

### 1. Clone & install

```bash
git clone https://github.com/bornfly-detachment/SubjectiveEgoneticsAI.git
cd SubjectiveEgoneticsAI

# Use the project's dedicated venv (LLaMA-Factory)
source ~/llama-factory/venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — fill in EGONETICS_TOKEN, ANTHROPIC_API_KEY, etc.
```

Key variables:

| Variable | Description |
|---|---|
| `EGONETICS_TOKEN` | Agent JWT from Egonetics (30-day) |
| `ANTHROPIC_API_KEY` | API key for LLM calls |
| `ANTHROPIC_BASE_URL` | Override for compatible endpoints (ARK, Kimi, etc.) |
| `DEFAULT_LLM_MODEL` | Model ID, e.g. `claude-sonnet-4-6` or `ark-code-latest` |
| `MODEL_PATH` | Path to local Qwen weights (for edge inference) |

> **HTTP Proxy**: If you need a proxy, set it at runtime — the port changes per session:
> ```bash
> export HTTP_PROXY=http://127.0.0.1:<port>
> ```

### 3. Initialize DB

```bash
python scripts/init_db.py
```

### 4. Start services

```bash
bash scripts/start.sh
```

This starts:
- **Main API** → `http://localhost:8000` (docs at `/docs`)
- **Edge inference** → `http://localhost:8001` (local Qwen model)

---

## API Reference

All endpoints under `http://localhost:8000`:

### Lifecycle

| Method | Path | Description |
|---|---|---|
| `POST` | `/lifecycle/start` | Translate task → graph, start agent loop |
| `POST` | `/lifecycle/stop/{task_id}` | Cancel running loop |
| `GET` | `/lifecycle/status/{task_id}` | Running state + trajectories + pending feedback |
| `POST` | `/lifecycle/feedback/{feedback_id}` | Resolve human-gate with `{user_response}` |
| `WS` | `/lifecycle/ws/{task_id}` | Real-time event stream |

**Start payload:**
```json
{ "task_id": "abc123" }
```
Optionally pass `"canvas_id"` to skip auto-translation and use an existing execution graph.

### WebSocket events

```
node_start       { node_id, canvas_id, node_kind }
node_complete    { node_id, canvas_id, cost }
node_failed      { node_id, canvas_id, error }
lifecycle_started
human_gate       { feedback_id, prompt }
feedback_resolved
```

### Agent, Feedback, Model

See `/docs` (Swagger UI) for full schema.

---

## Training Pipeline

SubjectiveEgoneticsAI learns from its own execution trajectories:

```
Task execution → trajectory saved → trigger threshold met
    → SFT fine-tune on (task, graph_json) pairs      [models/training/sft.py]
    → GRPO reinforcement on feedback signals          [models/training/grpo.py]
```

Constitution data (Notion-sourced) can be imported as SFT seeds:

```bash
python scripts/import_constitution.py
# Outputs: data/constitution/sft_alpaca.json, sft_judge.json
```

Training is via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — use the dedicated venv at `~/llama-factory/venv/`.

---

## Project Structure

```
SubjectiveEgoneticsAI/
├── api/
│   ├── main.py           FastAPI app, CORS, lifespan
│   ├── ws.py             WebSocket ConnectionManager
│   └── routes/
│       ├── agent.py      Agent session management
│       ├── feedback.py   Human feedback endpoints
│       ├── lifecycle.py  start/stop/status/ws
│       └── model.py      Model version management
├── agent/
│   ├── translator.py     LLM → execution graph
│   ├── executor.py       Node execution dispatch
│   ├── loop.py           DFS traversal + backtracking
│   └── backtrack.py      Failure strategy
├── modules/
│   ├── action.py         LLM API calls + tool dispatch
│   ├── judge.py          Output confidence scoring
│   ├── env_feedback.py   Environment feedback processing
│   └── agents/
│       ├── claude_code.py  Claude Code subprocess wrapper
│       └── openclaw.py     OpenClaw subprocess wrapper
├── models/
│   ├── inference.py      Local model FastAPI (vLLM-style)
│   └── training/
│       ├── sft.py         SFT via LLaMA-Factory
│       ├── grpo.py        GRPO reinforcement
│       ├── data_gen.py    Training data generation
│       └── versioning.py  Model version tracking
├── client/
│   └── egonetics.py      Egonetics REST client
├── config/
│   └── settings.py       Pydantic settings (reads .env)
├── store/
│   ├── db.py             SQLite connection + schema
│   └── schema.sql        DB schema
├── scripts/
│   ├── start.sh          Start all services
│   ├── init_db.py        Initialize SQLite
│   └── import_constitution.py  Notion → SFT data
├── tests/
├── .env.example
├── pyproject.toml
└── requirements.txt
```

---

## Design Principles

This project is built on **Egonetics** (Ego + Cybernetics):

- **Self-cybernetics**: The agent observes its own execution, scores outputs, and fine-tunes itself from trajectories
- **Human-in-the-loop**: Low-confidence nodes pause and emit `waiting_human` — the user can inject guidance via Egonetics `/agents` UI
- **Tamper-evident chronicle**: All decisions and outputs are written back to Egonetics as append-only page blocks
- **Modular agents**: Claude Code and OpenClaw are first-class worker agents, orchestrated by the same graph engine as local LLM nodes

---

## License

MIT
