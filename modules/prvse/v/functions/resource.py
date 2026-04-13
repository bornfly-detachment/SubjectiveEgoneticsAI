"""
V 层 — 资源消耗类 reward 函数
关注：token 效率、预算合规
"""
from modules.prvse.v.registry import reward_function
from modules.prvse.v.context import ExecutionContext


@reward_function(
    name="token_efficiency",
    description="Token 使用效率：实际用量占 budget 的比例越低越好",
    unit="score",
    weight=1.0,
    trigger="llm_call",
)
def token_efficiency(ctx: ExecutionContext) -> float:
    if ctx.failed:
        return 0.0
    total = ctx.token_total
    if total == 0:
        return 1.0
    budget = max(ctx.budget_tokens, 1)
    ratio = total / budget
    # <50% budget → 满分；50%~100% → 线性衰减；>100% → 负分
    if ratio <= 0.5:
        return 1.0
    elif ratio <= 1.0:
        return 1.0 - (ratio - 0.5) * 1.2   # 0.5 → 0.4
    else:
        return max(-1.0, 0.4 - (ratio - 1.0) * 0.8)


@reward_function(
    name="budget_compliance",
    description="是否在 budget 内完成：超出直接惩罚",
    unit="score",
    weight=0.8,
    trigger="llm_call",
)
def budget_compliance(ctx: ExecutionContext) -> float:
    if ctx.failed:
        return 0.0
    total = ctx.token_total
    budget = max(ctx.budget_tokens, 1)
    if total <= budget:
        return 1.0
    # 超出 budget 按比例惩罚
    overshoot = (total - budget) / budget
    return max(-1.0, 1.0 - overshoot * 2)
