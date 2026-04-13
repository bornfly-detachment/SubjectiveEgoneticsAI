"""
ExecutionContext — V 层 reward 函数统一入参
每次节点执行完毕后构建，传入所有已注册的 reward functions
"""
from dataclasses import dataclass, field


@dataclass
class ExecutionContext:
    task_id:        str
    trajectory_id:  str
    node_id:        str
    node_kind:      str           # 'llm_call'|'tool_call'|'local_judge'|'rule_branch'|'human_gate'|'entity'
    status:         str           # 'success'|'failed'
    input_context:  dict = field(default_factory=dict)
    output_result:  dict = field(default_factory=dict)
    cost_vector:    dict = field(default_factory=dict)   # token_input/token_output/time_ms/memory_mb
    net_time_ms:    int  = 0
    error_info:     str  = ""
    budget_tokens:  int  = 4000   # node-level token budget from exec_config

    # ── 便捷属性 ──────────────────────────────────────────────────────────────

    @property
    def token_total(self) -> int:
        return (self.cost_vector.get("token_input", 0) +
                self.cost_vector.get("token_output", 0))

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def failed(self) -> bool:
        return self.status == "failed"
