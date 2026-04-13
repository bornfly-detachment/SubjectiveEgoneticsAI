# Inference API

本地推理服务，基于 **Qwen3.5-0.8B + MLX**（Apple Silicon 加速）。

- 地址：`http://localhost:8001`
- 启动：`uvicorn models.inference:app --port 8001`
- 模型加载约 **30s**，启动时自动预热，后续请求无冷启动延迟
- 推理速度：**~20 tok/s**（M1，fp32）

---

## GET /health

确认服务与模型就绪状态，**每次发请求前先调用此接口**。

**响应**

```json
{
  "status": "ok",
  "model_loaded": true,
  "backend": "mlx"
}
```

| 字段 | 说明 |
|------|------|
| `status` | `"ok"` 表示服务正常 |
| `model_loaded` | `false` 时模型仍在加载，需等待 |
| `backend` | 当前推理引擎，固定为 `"mlx"` |

---

## POST /judge

**主推理接口**。输入问题与上下文，模型按宪法原则做主观判断，返回结构化结论。

### 请求体

```json
{
  "question": "这个行动是否符合诚实原则？",
  "context": {
    "action": "向用户隐瞒错误信息"
  },
  "constitution_hint": "个体应如实告知所知悉的事实"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `question` | string | ✓ | 需要判断的问题 |
| `context` | object | — | 背景上下文，任意 key-value，默认 `{}` |
| `constitution_hint` | string | — | 宪法原则提示，传入后模型会参考该约束 |

### 响应体

```json
{
  "answer": "否",
  "confidence": 0.85,
  "reasoning": "根据诚实原则，隐瞒错误信息会导致信息不对称，损害信任关系，并可能引发后续误解或错误决策。",
  "raw_output": "{\"answer\": \"否\", \"confidence\": 0.85, \"reasoning\": \"...\"}",
  "tokens_per_second": 19.8
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | `"是"` / `"否"` / `"不确定"` 三选一 |
| `confidence` | float | 模型自评置信度，范围 0.0 ~ 1.0 |
| `reasoning` | string | 推理依据说明 |
| `raw_output` | string | 模型原始输出（调试用） |
| `tokens_per_second` | float | 本次推理速度 |

### 置信度参考

| 值域 | 含义 | 前端建议 |
|------|------|---------|
| ≥ 0.8 | 高置信，判断可靠 | 直接展示结论 |
| 0.6 ~ 0.8 | 中等置信 | 展示结论 + reasoning |
| < 0.6 | 低置信，模型不确定 | 提示用户人工介入 |

> 系统阈值在 `config/settings.py` 的 `judge_confidence_threshold`（默认 0.6）控制。

---

## POST /generate

**通用文本生成接口**。自由输入 prompt，可选传入 system 角色，返回模型生成文本。

### 请求体

```json
{
  "prompt": "请用3句话解释什么是自我控制论",
  "system": "你是一位擅长简明解释的助手",
  "max_tokens": 512,
  "temperature": 0.7
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:----:|--------|------|
| `prompt` | string | ✓ | — | 用户输入 |
| `system` | string | — | `null` | System prompt，不传则无系统角色 |
| `max_tokens` | int | — | `512` | 最大生成 token 数 |
| `temperature` | float | — | `0.7` | 采样温度，越低越确定，越高越发散 |

### 响应体

```json
{
  "text": "自我控制论指出，人类行为并非天生固定，而是源于机体内部控制机制对抗外部诱惑的动态过程...",
  "tokens_per_second": 21.4
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 模型生成的完整文本 |
| `tokens_per_second` | float | 本次推理速度 |

---

## POST /reload-model

热切换模型权重，无需重启服务。训练完新 checkpoint 后使用。

### 请求

```
POST /reload-model?checkpoint_path=/path/to/checkpoint
```

### 响应

```json
{
  "status": "reloaded",
  "checkpoint": "/path/to/checkpoint"
}
```

---

## 前端接入示例

### 健康检查 + 推理（JS）

```js
const BASE = 'http://localhost:8001'

async function waitReady(maxRetries = 10) {
  for (let i = 0; i < maxRetries; i++) {
    const { model_loaded } = await fetch(`${BASE}/health`).then(r => r.json())
    if (model_loaded) return
    await new Promise(r => setTimeout(r, 3000))
  }
  throw new Error('Inference service not ready')
}

async function judge(question, context = {}, constitutionHint = null) {
  await waitReady()
  const res = await fetch(`${BASE}/judge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, context, constitution_hint: constitutionHint })
  }).then(r => r.json())

  // res.answer: "是" | "否" | "不确定"
  // res.confidence: 0.0 ~ 1.0
  // res.reasoning: string
  return res
}

async function generate(prompt, system = null, maxTokens = 512) {
  await waitReady()
  return fetch(`${BASE}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, system, max_tokens: maxTokens, temperature: 0.7 })
  }).then(r => r.json())
  // res.text: string
}
```

### 错误处理

服务仅在模型加载失败时返回 `500`，正常情况下：

- `/judge` 解析失败时 `answer` 为 `"不确定"`，`confidence` 为 `0.1`，不抛异常
- `/generate` 不做结构化校验，原样返回生成文本

---

## 注意事项

- 模型运行在本地 CPU/M1 GPU，单次 `/judge` 响应时间约 **1~3s**，前端需加 loading 状态
- 服务为**单进程单线程**，并发请求会排队，不支持并行推理
- `/reload-model` 期间服务不可用，约需 30s
