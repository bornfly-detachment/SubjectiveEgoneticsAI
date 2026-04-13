# SubjectiveEgoneticsAI — API 接口文档

**Base URL**: `http://localhost:8000`
**服务角色**: AI 执行内核，Egonetics（Node.js）通过 HTTP 调用本服务
**更新规则**: 每次新增/修改接口后同步更新本文档

---

## 服务状态

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回 `{"status": "ok"}` |

---

## Agent 轨迹 `/agent`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/agent/status/{task_id}` | 查询任务的执行轨迹（最近 30 条） |

**响应示例**
```json
{
  "task_id": "xxx",
  "trajectories": [
    { "id": "...", "task_id": "...", "node_id": "...", "cost_vector": "...", "reward": 0.8, "net_time_ms": 1200 }
  ]
}
```

---

## 任务生命周期 `/lifecycle`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/lifecycle/start` | 编译任务 + 启动 LangGraph Agent Loop |
| POST | `/lifecycle/stop/{task_id}` | 终止正在运行的任务 |
| GET  | `/lifecycle/status/{task_id}` | 任务状态 + 轨迹 + 待处理反馈 |
| POST | `/lifecycle/feedback/{feedback_id}` | 解除 human_gate 阻塞（恢复 LangGraph） |
| WS   | `/lifecycle/ws/{task_id}` | WebSocket 实时执行事件流 |

### POST `/lifecycle/start`
```json
// 请求体
{
  "task_id": "string",       // 必填
  "canvas_id": "string",     // 可选，为空则自动 NL→ExecGraph 编译
  "resources": {}            // 可选，额外资源
}
// 响应
{ "status": "started", "task_id": "...", "canvas_id": "..." }
```

### POST `/lifecycle/feedback/{feedback_id}`
```json
// 请求体
{ "user_response": "string" }
// 响应
{ "status": "resolved", "feedback_id": "...", "task_id": "..." }
```

### WebSocket `/lifecycle/ws/{task_id}`

连接后服务端主动推送事件，格式：
```json
{ "type": "event_name", "data": { ... } }
```

| 事件类型 | 触发时机 |
|----------|----------|
| `connected` | 连接成功，返回当前运行状态 |
| `compiling` | NL→ExecGraph 编译中 |
| `graph_ready` | 执行图就绪，开始运行 |
| `task_stopped` | 手动终止任务 |
| `task_failed` | 任务异常失败 |
| `feedback_resolved` | human_gate 反馈已解除 |

客户端可发送 `{"type": "ping"}` 保活，服务端回 `{"type": "pong"}`。

---

## 用户反馈 `/feedback`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST  | `/feedback/` | 创建反馈请求 |
| GET   | `/feedback/pending/{task_id}` | 查询任务的待处理反馈 |
| GET   | `/feedback/all` | 全部反馈列表（默认 limit=50） |
| PATCH | `/feedback/{feedback_id}/resolve` | 标记反馈为已解决 |
| GET   | `/feedback/failure-cases` | 查询失败案例（`?analyzed=false`） |
| PATCH | `/feedback/failure-cases/{case_id}` | 更新失败案例分析结果 |

### 反馈类型（`feedback_type`）
| 类型 | 说明 |
|------|------|
| `graph_update` | 执行图需要更新 |
| `failure_analysis` | 失败原因分析 |
| `decision_query` | 决策询问（阻塞） |
| `value_judgment` | 价值判断（阻塞） |

### POST `/feedback/`
```json
{
  "task_id": "string",
  "feedback_type": "decision_query",
  "context": {},
  "prompt": "是否继续执行？",
  "is_blocking": true
}
```

---

## 模型版本管理 `/model`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/model/versions` | 列出所有模型版本（checkpoint） |
| GET  | `/model/active` | 当前激活的模型版本 |
| POST | `/model/activate/{version_id}` | 激活指定版本（自动热替换推理服务） |
| GET  | `/model/training-status` | 检查 SFT/GRPO 触发条件 |
| POST | `/model/train/sft` | 触发 SFT 训练（后台异步） |
| POST | `/model/train/grpo` | 触发 GRPO 训练（后台异步） |

```json
// GET /model/training-status 响应
{ "sft_ready": false, "grpo_ready": true }
```

---

## LLM 对话 `/llm`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/llm/chat` | 多轮对话，支持同步和 SSE 流式输出 |

### POST `/llm/chat`
```json
// 请求体
{
  "messages":   [{"role": "user", "content": "你好"}],
  "system":     "你是一个助手",   // 可选
  "model":      "MiniMax-M2.7", // 可选，默认 settings.default_llm_model
  "max_tokens": 2048,              // 可选
  "stream":     false              // true = SSE 流式
}

// 同步响应
{ "text": "...", "usage": { "input_tokens": 10, "output_tokens": 50 } }

// 流式响应（SSE）
data: {"type": "delta", "text": "片"}
data: {"type": "delta", "text": "段"}
data: {"type": "done",  "usage": {"input_tokens": 10, "output_tokens": 50}}
```

---

## PRVSE 控制论内核

### V 层（Reward Functions）`/prvse/v` ✅

| 方法 | 路径 | 说明 |
|------|------|------|
| GET   | `/prvse/v/functions` | 所有注册函数 + 当前权重 |
| PATCH | `/prvse/v/functions/{id}/weight` | 调整权重 `{weight: float}` — 热更新+持久化 |
| POST  | `/prvse/v/compute` | 手动触发 reward 计算 |
| GET   | `/prvse/v/history/{task_id}` | 任务的 reward 历史 |
| GET   | `/prvse/v/stats` | 近期 reward 统计（total/avg/min/max/high/low） |

**已注册 reward functions（8个）**：
| 函数 | trigger | weight | 说明 |
|------|---------|--------|------|
| `token_efficiency` | llm_call | 1.0 | Token 使用效率 |
| `budget_compliance` | llm_call | 0.8 | 预算合规 |
| `task_success` | any | 2.0 | 节点执行成功 |
| `output_completeness` | llm_call | 0.6 | 输出非空有内容 |
| `execution_speed` | any | 0.5 | 执行速度 |
| `timeout_penalty` | any | 1.2 | 超时惩罚 |
| `error_free` | any | 1.5 | 无报错 |
| `tool_call_success` | tool_call | 1.0 | 工具调用成功 |

权重由 Egonetics V 层面板（`/cybernetics` → V 层 Reward tab）人工调整，持久化到 `v_function_weights` 表。
每次 AgentLoop 节点执行后自动计算 reward 写入 `trajectories.reward`。
### E0 生命周期 + PRVSE 组件树 `/e0` ✅

#### 生命周期 CRUD

| 方法 | 路径 | 说明 |
|------|------|------|
| GET    | `/e0/lifecycles` | 列出所有生命周期（e0/task/agent/model + 用户自定义） |
| POST   | `/e0/lifecycles` | 新建自定义生命周期 `{id, name, description}` |
| PATCH  | `/e0/lifecycles/{id}` | 编辑 `{name?, description?, enabled?}` |
| DELETE | `/e0/lifecycles/{id}` | 删除（仅用户自定义，内置用 `enabled=false` 禁用） |
| GET    | `/e0/lifecycles/{id}/state` | 当前状态 + 合法转换列表 |
| POST   | `/e0/lifecycles/{id}/state/transition` | 触发状态转换 `{to_state, meta?}` |

**内置生命周期**：`e0`（全局）、`task`、`agent`、`model`

**状态机合法转换**：
```
IDLE → OBSERVING → REFLECTING → TRAINING → VALIDATING → ACTIVATING → IDLE
                                                  ↓
                                            REFLECTING  （回退）
```

#### PRVSE 组件树 CRUD

| 方法 | 路径 | 说明 |
|------|------|------|
| GET    | `/e0/components` | 列出组件（`?lifecycle_id=&layer=` 过滤） |
| POST   | `/e0/components` | 新建 `{lifecycle_id, layer, sub_id, name, description?, config?}` |
| PATCH  | `/e0/components/{id}` | 编辑 `{name?, description?, status?, config?}` |
| DELETE | `/e0/components/{id}` | 删除（仅用户自定义） |

**status 合法值**：`active` / `inactive` / `error` / `running`
**component id 格式**：`{lifecycle_id}.{layer}.{sub_id}`（例：`e0.P.observe`）

内置 21 个组件（仅 e0 生命周期）：P×4 / R×4 / V×5 / S×3 / E×5

| S 层（Relation Inference） | `/prvse/r` | 计划中 |
| P 层（Information Classifier） | `/prvse/p` | 计划中 |
| E 层（Global Optimizer） | `/prvse/e` | 计划中 |

---

## 内部模块能力（不直接暴露为 HTTP，供路由层调用）

### ActionModule (`modules/action.py`)
| 方法 | 说明 |
|------|------|
| `llm_call(prompt, system, context, model, max_tokens)` | 调用 LLM（支持 Anthropic/OpenAI 协议），返回 `{content, input_tokens, output_tokens}` |
| `tool_call(tool_name, args, context)` | 调用工具 |

**已注册工具**：`read_file` / `write_file` / `search_web`（未配置）/ `run_python`（未启用）/ `claude_code` / `openclaw`

### JudgeModule (`modules/judge.py`)
置信度评估，决定是否需要用户介入。

### AgentLoop (`agent/loop.py`)
LangGraph 执行引擎，驱动 ExecGraph 节点按依赖顺序执行。

### Translator (`agent/translator.py`)
自然语言任务描述 → ExecGraph（Canvas）编译。

---

## LLM 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `default_llm_provider` | `anthropic` | `anthropic` 或 `openai` |
| `default_llm_model` | `MiniMax-M2.7` | MiniMax API，兼容 Anthropic SDK |
| `anthropic_base_url` | `https://api.minimaxi.com/anthropic` | MiniMax endpoint |
| `llm_proxy` | `""` | HTTP 代理（端口会变，运行时 `export HTTP_PROXY=...`） |

---

## Egonetics 调用点汇总

| Egonetics 页面/组件 | 调用方式 | 目标接口 |
|---------------------|----------|----------|
| `/agents` E0 控制台 | 直连 `:8000` | `/e0/lifecycles`, `/e0/components`, `/lifecycle/*`, WS `/lifecycle/ws/:taskId` |
| `/cybernetics` V 层面板 | 直连 `:8000` | `/prvse/v/*` |
| 右下角 LLMChatDialog | 经 Egonetics `:3002` 转发 | 待迁移至 `/llm/chat` |
