from __future__ import annotations

from datetime import datetime
import ast
import operator
import uuid
from typing import Any, Dict


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


def get_date() -> str:
    """获取当前日期字符串。

    返回格式：YYYY-MM-DD
    示例：2025-01-31
    """
    return datetime.now().strftime("%Y-%m-%d")


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_expression(expression: str) -> float:
    """安全计算四则运算表达式。

    仅允许数字、+ - * / ** // % 和括号。
    """
    if not expression or len(expression) > 200:
        raise ValueError("表达式为空或长度过长。")

    parsed = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            return _ALLOWED_BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            return _ALLOWED_UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError("表达式包含不支持的元素。")

    return _eval(parsed)


def make_uuid() -> str:
    """生成 UUID4 字符串。"""
    return str(uuid.uuid4())


def text_stats(text: str) -> Dict[str, Any]:
    """返回文本的基础统计信息。"""
    stripped = text.strip()
    return {
        "chars": len(text),
        "chars_no_space": len(stripped.replace(" ", "")),
        "words": len([part for part in stripped.split() if part]),
    }
