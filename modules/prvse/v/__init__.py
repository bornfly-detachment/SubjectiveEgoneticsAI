"""
modules/prvse/v — V 层公开接口

导入此包时自动触发所有 @reward_function 的注册。
"""
# 触发注册：导入所有 functions 模块
from modules.prvse.v.functions import resource, quality, time, reliability  # noqa: F401

from modules.prvse.v.registry import (
    VRegistry,
    reward_function,
    compute_reward,
    RewardResult,
)
from modules.prvse.v.context import ExecutionContext

__all__ = [
    "VRegistry",
    "reward_function",
    "compute_reward",
    "RewardResult",
    "ExecutionContext",
]
