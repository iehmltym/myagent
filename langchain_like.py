"""LangChain 核心思想的超轻量实现。

这个模块不依赖 LangChain，用最少的代码模拟其“组合式”设计：
- Prompt 模板
- Runnable/LCEL 管道
- 输出解析器

这样做的目的不是替代 LangChain，而是帮助你理解核心概念。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Protocol
import json


class Runnable(Protocol):
    """最小化 Runnable 协议：可被链式调用的执行单元。"""

    def invoke(self, input_data: Any) -> Any:
        """同步执行并返回结果。"""

    def batch(self, inputs: Iterable[Any]) -> List[Any]:
        """批量执行，默认逐个调用。"""
        return [self.invoke(item) for item in inputs]

    def __or__(self, other: "Runnable") -> "Runnable":
        """支持使用 | 运算符组合 Runnable。"""
        return RunnableSequence([self, other])


class RunnableSequence:
    """串行执行多个 Runnable，形成 LCEL 风格的流水线。"""

    def __init__(self, steps: List[Runnable]):
        self.steps = steps

    def invoke(self, input_data: Any) -> Any:
        result = input_data
        for step in self.steps:
            result = step.invoke(result)
        return result

    def batch(self, inputs: Iterable[Any]) -> List[Any]:
        results = []
        for item in inputs:
            results.append(self.invoke(item))
        return results

    def __or__(self, other: Runnable) -> "RunnableSequence":
        return RunnableSequence(self.steps + [other])


@dataclass
class PromptTemplate(Runnable):
    """基础 Prompt 模板。

    使用 format_map 将变量注入模板中。
    """

    template: str

    def invoke(self, input_data: Dict[str, Any]) -> str:
        return self.template.format_map(input_data)


@dataclass
class ChatPromptTemplate(Runnable):
    """模拟 ChatPromptTemplate，输出可读的对话格式。"""

    messages: List[Dict[str, str]]

    def invoke(self, input_data: Dict[str, Any]) -> str:
        rendered: List[str] = []
        for msg in self.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").format_map(input_data)
            rendered.append(f"{role.upper()}: {content}")
        return "\n".join(rendered)


@dataclass
class FunctionRunnable(Runnable):
    """把任意函数包装成 Runnable。"""

    fn: Callable[[Any], Any]

    def invoke(self, input_data: Any) -> Any:
        return self.fn(input_data)


class OutputParser(Runnable):
    """输出解析器基类。"""

    def invoke(self, input_data: Any) -> Any:
        return self.parse(input_data)

    def parse(self, text: str) -> Any:
        raise NotImplementedError


class JsonOutputParser(OutputParser):
    """解析 JSON 字符串，适合结构化输出。"""

    def parse(self, text: str) -> Any:
        return json.loads(text)


@dataclass
class PydanticOutputParser(OutputParser):
    """把 JSON 字符串解析成 Pydantic 模型。"""

    model_cls: Any

    def parse(self, text: str) -> Any:
        data = json.loads(text)
        return self.model_cls(**data)


@dataclass
class LLMRunnable(Runnable):
    """把任意 LLM 客户端封装成 Runnable。"""

    llm: Any
    llm_kwargs: Dict[str, Any] | None = None

    def invoke(self, input_data: str) -> str:
        kwargs = self.llm_kwargs or {}
        return self.llm.generate(input_data, **kwargs)
