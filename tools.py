from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from datetime import datetime
from io import BytesIO
import operator
import uuid
from typing import Any, Dict
import wave


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


def normalize_whitespace(text: str) -> str:
    """压缩多余空白并去除首尾空格。"""
    return " ".join(text.split())


def slugify(text: str) -> str:
    """将文本转为 URL slug。"""
    lowered = text.strip().lower()
    slug = re.sub(r"[^a-z0-9\-\s]", "", lowered)
    slug = re.sub(r"[\s\-]+", "-", slug)
    return slug.strip("-")


def extract_urls(text: str) -> Dict[str, Any]:
    """提取文本中的 URL。"""
    urls = re.findall(r"https?://[^\s)\]]+", text)
    return {"count": len(urls), "items": urls}


def extract_emails(text: str) -> Dict[str, Any]:
    """提取文本中的邮箱地址。"""
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return {"count": len(emails), "items": emails}


def markdown_outline(text: str) -> Dict[str, Any]:
    """提取 Markdown 标题大纲。"""
    outline = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if match:
            outline.append({"level": len(match.group(1)), "title": match.group(2).strip()})
    return {"count": len(outline), "items": outline}


def dedupe_lines(text: str) -> str:
    """按行去重，保留首次出现顺序。"""
    seen = set()
    unique_lines = []
    for line in text.splitlines():
        if line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)
    return "\n".join(unique_lines)


def top_keywords(text: str, top_n: int = 6) -> Dict[str, Any]:
    """统计高频关键词。"""
    words = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    filtered = [word for word in words if len(word) > 1]
    counts: Dict[str, int] = {}
    for word in filtered:
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "count": len(ranked),
        "items": [
            {"word": word, "count": count} for word, count in ranked[: max(1, top_n)]
        ],
    }


def json_prettify(text: str) -> str:
    """格式化 JSON 字符串。"""
    payload = json.loads(text)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def estimate_tokens(text: str) -> Dict[str, Any]:
    """粗略估算 token 数量。"""
    est = math.ceil(len(text) / 4)
    return {"chars": len(text), "estimated_tokens": est}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text_preview(data: bytes, limit: int = 1200) -> str:
    if not data:
        return ""
    decoded = data.decode("utf-8", errors="replace")
    return decoded[:limit]


def describe_file(data: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """返回文件的基础描述信息。"""
    return {
        "filename": filename,
        "content_type": content_type,
        "size_bytes": len(data),
        "sha256": _sha256(data),
        "text_preview": _text_preview(data),
    }


def _png_size(data: bytes) -> Dict[str, int]:
    if len(data) < 24:
        return {}
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return {}
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return {"width": width, "height": height}


def _gif_size(data: bytes) -> Dict[str, int]:
    if len(data) < 10:
        return {}
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")
    return {"width": width, "height": height}


def _jpeg_size(data: bytes) -> Dict[str, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return {}
    idx = 2
    while idx < len(data) - 1:
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        if marker in {0xC0, 0xC2}:
            if idx + 8 >= len(data):
                return {}
            height = int.from_bytes(data[idx + 5 : idx + 7], "big")
            width = int.from_bytes(data[idx + 7 : idx + 9], "big")
            return {"width": width, "height": height}
        if idx + 3 >= len(data):
            break
        seg_length = int.from_bytes(data[idx + 2 : idx + 4], "big")
        if seg_length < 2:
            break
        idx += 2 + seg_length
    return {}


def _detect_image_kind(data: bytes) -> str:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 6 and data[:6] in {b"GIF87a", b"GIF89a"}:
        return "gif"
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        return "jpeg"
    return "unknown"


def describe_image(data: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """返回图片的基础描述信息。"""
    kind = _detect_image_kind(data)
    size = {}
    if kind == "png":
        size = _png_size(data)
    elif kind == "gif":
        size = _gif_size(data)
    elif kind == "jpeg":
        size = _jpeg_size(data)
    return {
        "filename": filename,
        "content_type": content_type,
        "format": kind,
        "size_bytes": len(data),
        "sha256": _sha256(data),
        **size,
    }


def describe_audio(data: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    """返回音频基础信息（wav 可解析时长）。"""
    duration = None
    try:
        with wave.open(BytesIO(data)) as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = round(frames / float(rate), 2) if rate else None
    except wave.Error:
        duration = None
    return {
        "filename": filename,
        "content_type": content_type,
        "size_bytes": len(data),
        "sha256": _sha256(data),
        "duration_seconds": duration,
    }
