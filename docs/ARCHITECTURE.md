# Egonetics × SubjectiveEgoneticsAI — 系统架构文档

**版本**: v1
**最后对齐**: 2026-03-24
**维护规则**: 每次架构决策变更后同步更新本文档

---

## 一、双项目职责边界（硬性规则）

```
Egonetics (Node.js · port 3000/3002)
  ├── 展示层：UI、数据可视化、用户交互
  ├── 人机协同：human gate、反馈输入、PRVSE 树编辑
  └── 禁止：实现 AI/Agent/推理/训练逻辑

SubjectiveEgoneticsAI (Python · port 8000)
  ├── 执行内核：控制论、Agent 调度、PRVSE 实现
  ├── 本地模型训练推理（LlamaFactory / GRPO / SFT）
  └── 禁止：处理前端路由、用户认证（走 Egonetics）

通信：Egonetics → HTTP / WebSocket → SEAI:8000
```

**开发态是深度耦合的**：Egonetics 和 SEAI 是一盘棋，共同打磨人机协同迭代模式。
**生产态才是松耦合**：SEAI 可独立部署，Egonetics 通过 HTTP 调用。

---

## 二、开发铁律（最高优先级，无例外）

> **任何数据结构、配置、规则、参数的实现，必须同时满足 CRUD。**

- 后端：必须有 `GET / POST / PATCH / DELETE` 路由
- 前端：用户必须能在界面上增删改查，不能只读
- **不满足 CRUD 的功能，不做。宁可不实现，不做残缺品。**

动手前必须回答：用户如何 Create / Read / Update / Delete 这里的数据？答不上来 → 不做。

**违规案例（已发生，不得重犯）**：
- V 层 reward functions 用 `@reward_function` 装饰器硬编码 → 用户无法增删改 → 已修复为 DB 驱动

---

## 三、PRVSE 控制论框架

PRVSE 是系统的**控制论骨架**。每个字母是一个独立的 AOP 层，按职责切面注入，不侵入业务逻辑。

```
P — Perception  感知层    收集所有输入，分类、检测、压缩原始信号
R — Relation    关系层    实体识别、因果链接、图谱推断
V — Value       价值层    多维价值评估（5D 向量，非标量）
S — State       状态层    状态定义、状态机转换、生命周期管理
E — Evolution   进化层    Diff 收集、训练触发、模型迭代
```

### AOP 实现模式

```python
# @reward_function = AOP 注解
# VRegistry       = AOP 织入器
# 在 agent node 执行完成后自动触发，不侵入业务逻辑

@reward_function(name="task_success", weight=2.0, trigger="any")
def task_success(ctx: ExecutionContext) -> float:
    return 1.0 if ctx.succeeded else -0.5
```

---

## 四、PRVSE 完整组件树

```
P — 感知层（所有输入的接收与分类）
├── P.observe    观察：收集执行轨迹、反馈、用户行为
├── P.classify   分类：判断信息归属类型
├── P.detect     检测：发现异常、重复模式、偏差
└── P.compress   压缩：历史摘要，减少信息熵

R — 关系层（实体间连接与推断）
├── R.entity     实体注册：task / agent / model / user
├── R.link       连接：建立实体间的关系边
├── R.infer      推断：从已有关系自动派生新关系
└── R.graph      图谱：维护全局关系网络

V — 价值层（任何信息或行动的价值评估）
├── V.local      对当前局部目标的价值  [-1, 1]
├── V.global     对系统全局目标的价值  [-1, 1]
├── V.now        当前立即可用程度      [0, 1]
├── V.future     未来积累价值          [0, 1]
└── V.certainty  以上判断的确定性      [0, 1]

S — 状态层（所有实体的生命周期管理）
├── S.define     状态空间定义（哪些状态存在）
├── S.transition 转移规则（什么条件触发什么转移）
└── S.lifecycle  生命周期实例
    ├── S.lifecycle.e0      E0 系统自身（核心）
    │   IDLE → OBSERVING → REFLECTING → TRAINING → VALIDATING → ACTIVATING
    ├── S.lifecycle.task    任务生命周期
    ├── S.lifecycle.agent   Agent 生命周期
    └── S.lifecycle.model   模型版本生命周期

E — 进化层（差分驱动的自我改进）
├── E.diff       差分收集：AI 输出 vs 真实/人工修正
├── E.trigger    触发判断：满足条件时启动训练
├── E.train      训练执行：SFT / GRPO
├── E.validate   有效性验证：before/after 指标对比
└── E.activate   模型切换：激活新版本 / 回滚
```

---

## 五、V 层 — 价值向量（5D，非标量）

### 定义

```python
V = {
    local:     float  # [-1, 1]  单次执行局部收益
    global:    float  # [-1, 1]  对整体目标的贡献
    now:       float  # [0,  1]  立即收益
    future:    float  # [0,  1]  长期潜力
    certainty: float  # [0,  1]  价值判断的置信度
}
```

**为什么是向量而不是标量**：标量 reward 无法区分"短期好但长期差"、
"局部成功但全局无意义"，这正是 AI 对齐失控的根源。
V 的五个维度让训练信号有方向感。

### 当前实现状态

| 状态 | 说明 |
|------|------|
| ✅ 已实现 | 标量 `total_reward`（各函数加权求和），DB `trajectories.reward` |
| ⏳ 待重构 | 每个 reward function 对应具体维度；DB 增 `reward_vector JSON` 列 |

### Reward Functions

- DB 表：`v_functions`（`is_builtin` 区分系统内置 / 用户自定义）
- 系统内置：Python 函数，可调权重/开关，不可修改表达式
- 用户自定义：`expression` 字段存 Python 表达式，受限 `eval` 执行
- API：`GET/POST/PATCH/DELETE /prvse/v/functions`
- 前端：`/cybernetics` → "V 层 Reward" tab

---

## 六、E0 生命周期状态机

### E0 是什么

**永动环**——AI 挂了环继续转，数据继续积累，AI 恢复后接着跑。
控制流是确定性的（机器执行），AI 是填充物（随训练变强）。

```
[收集 Diff] → [积累判断] → [触发训练]
      ↑                          ↓
[健康检查] ← [模型注册表] ← [训练 + 验证]
```

### 状态机

```
IDLE → OBSERVING → REFLECTING → TRAINING → VALIDATING → ACTIVATING → IDLE
                                                 ↓
                                           REFLECTING  （验证失败回退）
```

| 状态 | 含义 |
|------|------|
| IDLE | 待机，无主动行为 |
| OBSERVING | 收集 trajectories、feedback、diff |
| REFLECTING | 分析数据，识别改进方向 |
| TRAINING | 执行 GRPO/SFT 训练 |
| VALIDATING | 验证新模型性能，与 baseline 对比 |
| ACTIVATING | A/B 测试通过，切换活跃模型 |

### 合法转换规则

```python
VALID_TRANSITIONS = {
    "IDLE":       ["OBSERVING"],
    "OBSERVING":  ["REFLECTING", "IDLE"],
    "REFLECTING": ["TRAINING", "IDLE"],
    "TRAINING":   ["VALIDATING", "IDLE"],
    "VALIDATING": ["ACTIVATING", "REFLECTING"],  # REFLECTING = 回退
    "ACTIVATING": ["IDLE"],
}
```

### 内置生命周期

| id | name | 用途 |
|----|------|------|
| `e0` | E0 全局 | 系统级自进化主循环 |
| `task` | Task | 单任务执行生命周期 |
| `agent` | Agent | Agent 实例生命周期 |
| `model` | Model | 本地模型训练/服务生命周期 |

---

## 七、E 层 — Diff 驱动训练

### Diff 数据结构（已对齐，待实现）

```python
Diff = {
    id:           str
    diff_type:    "v_assessment"   # AI 对 V 的判断错了
                | "action_choice"  # AI 选了错的行动
                | "prediction"     # AI 预测和实际不符
                | "goal_interpret" # AI 误解了目标

    context:      dict   # 当时的 ExecutionContext 快照
    ai_output:    dict   # AI 给出的内容（V 向量 / 行动 / 预测）
    ground_truth: dict   # 真实发生的 / 人类修正的

    magnitude:    float  # 偏差大小 [0,1]，决定是否值得训练
    v_delta:      dict   # V 评估前后的差值（知道错在哪个维度）

    trainable:    bool   # 是否可用于训练（人工标记或自动判断）
    trained:      bool
    created_at:   datetime
}
```

### 训练触发条件（规则本身也必须 CRUD）

```
magnitude_avg > 0.3  AND  diff_count > 50  →  触发 SFT（快速修正）
magnitude_avg > 0.5  AND  diff_count > 20  →  触发 GRPO（强化）
human 手动标记                             →  立即触发
```

触发规则存 DB，用户可在 Egonetics 界面增删改。

### 如何判断训练有效

```
1. 保留 20% Diff 不训练（验证集）
2. 训练后在验证集重放：新模型 magnitude 均值 < 旧模型 × 0.8
3. 在线 A/B：新模型 trajectory.reward 均值 ≥ 旧模型 × 0.95
→ 通过则激活（ACTIVATING），不通过则回滚（VALIDATING → REFLECTING）
```

---

## 八、本地模型训练边界

| 训练目标 | 对应 Diff 类型 | 验证方式 | 是否训练 |
|----------|----------------|----------|----------|
| V 评估（5D 向量） | `v_assessment` | 和真实 V 的 MSE 下降 | ✅ |
| 情境分类（P 层） | `goal_interpret` | 分类准确率提升 | ✅ |
| 简单决策分支 | `action_choice`（确定性场景） | 回放准确率 | ✅ |
| 复杂推理 | — | — | ❌ 永远是大模型 |
| 代码生成 | — | — | ❌ 永远是大模型 |
| 新颖/未见问题 | — | — | ❌ 大模型 fallback |

**本地小模型**（Qwen2）：快速廉价判断，逐步替代大模型的可确定性部分
**大模型**（ARK·Claude）：复杂推理、规划、few-shot，永远不训练

训练环境：`~/llama-factory/venv`（不得使用 conda base 或其他 venv）

---

## 九、数据库结构总览（SEAI）

| 表名 | 用途 |
|------|------|
| `trajectories` | 每次节点执行记录 + scalar reward |
| `cost_records` | 执行成本明细（time/memory/token） |
| `user_feedback` | 用户反馈（4 种类型） |
| `failure_cases` | 失败案例 + SFT/GRPO 训练样本 |
| `model_versions` | 模型版本管理 + AB 测试结果 |
| `constitution_rules` | 宪法规则（人类语言 → 机器规则） |
| `v_functions` | V 层 reward functions（CRUD） |
| `e0_lifecycles` | E0 状态机（4 个内置生命周期） |
| `prvse_components` | PRVSE 组件树（21 个内置节点） |
| *(待建)* `diffs` | Diff 队列 |
| *(待建)* `train_triggers` | 训练触发规则 |

---

## 十、前端页面与功能对应

| 路由 | 组件 | 功能 |
|------|------|------|
| `/agents` | `AgentsView.tsx` | E0 控制台：状态机 + PRVSE 树 CRUD |
| `/cybernetics` → V tab | `SEAIVPanel.tsx` | V 层 reward functions CRUD + 统计 |
| `/tasks` | `KanbanBoard.tsx` | 任务看板（勿动） |
| `/tasks/:id` | `TaskPageView.tsx` | 任务详情（勿动） |
| `/blog` | `PageManager.tsx` | 页面编辑器（勿动） |

---

## 十一、API 速查（SEAI · `:8000`）

```
GET  /health

# V 层 Reward Functions (CRUD)
GET    /prvse/v/functions
POST   /prvse/v/functions
PATCH  /prvse/v/functions/{id}
DELETE /prvse/v/functions/{id}
POST   /prvse/v/compute
GET    /prvse/v/history/{task_id}
GET    /prvse/v/stats

# E0 生命周期 (CRUD)
GET    /e0/lifecycles
POST   /e0/lifecycles
PATCH  /e0/lifecycles/{id}
DELETE /e0/lifecycles/{id}
GET    /e0/lifecycles/{id}/state
POST   /e0/lifecycles/{id}/state/transition     body: {to_state, meta?}

# PRVSE 组件树 (CRUD)
GET    /e0/components                           ?lifecycle_id=&layer=
POST   /e0/components
PATCH  /e0/components/{id}
DELETE /e0/components/{id}

# Agent 任务执行
GET    /agent/status/{task_id}
POST   /lifecycle/start                         body: {task_id}
POST   /lifecycle/stop/{task_id}
GET    /lifecycle/status/{task_id}
WS     /lifecycle/ws/{task_id}
POST   /lifecycle/feedback/{fb_id}

# LLM
POST   /llm/chat                                body: {messages, system?, model?, stream?}
```

---

## 十二、实现状态与下一步

### 已实现 ✅

| 模块 | 位置 |
|------|------|
| V 层 Reward Functions CRUD + AOP 织入 | `modules/prvse/v/` + `/prvse/v/*` |
| E0 生命周期状态机 CRUD | `modules/prvse/s/lifecycle.py` + `/e0/*` |
| PRVSE 组件树 CRUD（21 节点） | DB `prvse_components` + `/e0/components` |
| Agent 执行 + trajectory 记录 + reward 写入 | `agent/nodes.py` |
| LLM Chat API | `api/routes/llm.py` |
| E0 控制台前端 | `AgentsView.tsx` |
| V 层面板前端 | `SEAIVPanel.tsx` |

### 待实现（优先级排序）

| 优先级 | 模块 | 说明 |
|--------|------|------|
| 🔴 P0 | **E.diff 实现** | Diff 数据结构 + 收集入口（nodes.py + human_gate） |
| 🔴 P0 | **E.trigger 实现** | 触发规则 CRUD + 后台 watcher |
| 🟡 P1 | **V 向量化** | reward functions 按 5D 重组，DB 增 `reward_vector` |
| 🟡 P1 | **E.train + E.validate** | GRPO/SFT 执行 + 验证集回放 |
| 🟢 P2 | **Task/Agent/Model 生命周期组件** | 为另外 3 个生命周期补充 PRVSE 树 |
| 🟢 P2 | **PRVSE 组件激活逻辑** | 组件 `status=active` 时触发实际后端 hook |
