"""
V 层 — 输出质量类 reward 函数
关注：任务成功、输出完整性
"""
from modules.prvse.v.registry import reward_function
from modules.prvse.v.context import ExecutionContext


@reward_function(
    name="task_success",
    description="节点执行成功：成功得满分，失败得零分",
    unit="score",
    weight=2.0,     # 权重最高，成功是基本前提
    trigger="any",
)
def task_success(ctx: ExecutionContext) -> float:
    return 1.0 if ctx.succeeded else 0.0


@reward_function(
    name="output_completeness",
    description="输出结果非空且有内容",
    unit="score",
    weight=0.6,
    trigger="llm_call",
)
def output_completeness(ctx: ExecutionContext) -> float:
    if ctx.failed:
        return 0.0
    result = ctx.output_result
    if not result:
        return 0.0
    # 如果 output 是字符串内容
    if isinstance(result, str):
        return 1.0 if len(result.strip()) > 10 else 0.3
    # 如果是 dict，检查 content 字段
    content = result.get("content", result.get("text", ""))
    if isinstance(content, str):
        return 1.0 if len(content.strip()) > 10 else 0.3
    # 其他结构视为有效
    return 0.8
