"""观测与调试相关的轻量实现。

提供：
- TraceSpan：记录执行步骤耗时
- TraceStore：保存最近的 trace 结果
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class TraceSpan:
    """记录一个步骤的耗时和输入输出摘要。"""

    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)

    def finish(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class TraceStore:
    """保存最近一次请求的 trace。"""

    def __init__(self) -> None:
        self.spans: List[TraceSpan] = []

    def start_span(self, name: str, inputs: Dict[str, str] | None = None) -> TraceSpan:
        span = TraceSpan(name=name, inputs=inputs or {})
        self.spans.append(span)
        return span

    def finish_span(self, span: TraceSpan, outputs: Dict[str, str] | None = None) -> None:
        if outputs:
            span.outputs = outputs
        span.finish()

    def export(self) -> List[Dict[str, str | float]]:
        payload: List[Dict[str, str | float]] = []
        for span in self.spans:
            payload.append(
                {
                    "name": span.name,
                    "duration_ms": span.duration_ms,
                    "inputs": str(span.inputs),
                    "outputs": str(span.outputs),
                }
            )
        return payload

    def clear(self) -> None:
        self.spans.clear()
