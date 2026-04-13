"""
V 层 — 可靠性类 reward 函数
关注：无错误执行、工具调用成功率
"""
from modules.prvse.v.registry import reward_function
from modules.prvse.v.context import ExecutionContext


@reward_function(
    name="error_free",
    description="执行无报错：有 error_info 则惩罚",
    unit="score",
    weight=1.5,
    trigger="any",
)
def error_free(ctx: ExecutionContext) -> float:
    return 0.0 if ctx.error_info else 1.0


@reward_function(
    name="tool_call_success",
    description="工具调用成功率：output 中 ok=True",
    unit="score",
    weight=1.0,
    trigger="tool_call",
)
def tool_call_success(ctx: ExecutionContext) -> float:
    if ctx.failed:
        return 0.0
    result = ctx.output_result
    if isinstance(result, dict):
        ok = result.get("ok")
        if ok is True:
            return 1.0
        if ok is False:
            return 0.0
    # 无明确 ok 字段但执行成功，视为部分分
    return 0.7 if ctx.succeeded else 0.0
