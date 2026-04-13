"""
V 层 — 时间效率类 reward 函数
关注：执行速度、超时惩罚
"""
from modules.prvse.v.registry import reward_function
from modules.prvse.v.context import ExecutionContext

_TIMEOUT_MS = 300_000   # 5 分钟，对应 settings.node_timeout_seconds=300


@reward_function(
    name="execution_speed",
    description="执行速度：耗时越短得分越高",
    unit="score",
    weight=0.5,
    trigger="any",
)
def execution_speed(ctx: ExecutionContext) -> float:
    ms = ctx.net_time_ms
    if ms <= 0:
        return 1.0
    ratio = ms / _TIMEOUT_MS
    # <10% 超时阈值 → 满分；线性衰减到 100% 时为 0
    return max(0.0, 1.0 - ratio)


@reward_function(
    name="timeout_penalty",
    description="接近超时时强惩罚（>80% timeout 阈值）",
    unit="score",
    weight=1.2,
    trigger="any",
)
def timeout_penalty(ctx: ExecutionContext) -> float:
    ms = ctx.net_time_ms
    if ms <= 0:
        return 1.0
    ratio = ms / _TIMEOUT_MS
    if ratio < 0.8:
        return 1.0
    # 80%~100% 线性惩罚
    if ratio <= 1.0:
        return 1.0 - (ratio - 0.8) * 5    # 0.8→1.0，0.0
    # 超时直接负分
    return max(-1.0, -(ratio - 1.0) * 2)
