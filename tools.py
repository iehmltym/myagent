from __future__ import annotations

from datetime import datetime


def get_time() -> str:
    """获取当前时间字符串。

    返回格式：YYYY-MM-DD HH:MM:SS
    示例：2025-01-31 09:30:00
    """
    # datetime.now() 会获取系统当前的本地时间（不是 UTC）
    now = datetime.now()
    # strftime 用指定格式把时间对象转成字符串，便于直接返回给前端
    return now.strftime("%Y-%m-%d %H:%M:%S")


def add(a: float, b: float) -> float:
    """执行加法并返回结果。

    这里接收 float，既可以处理整数，也可以处理小数。
    """
    # Python 的 + 会进行数值相加
    return a + b
