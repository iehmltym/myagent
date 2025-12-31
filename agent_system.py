"""工具调用与 Agent 的轻量实现。

包含：
- Tool 定义（输入 schema、执行函数）
- 简单 Agent 路由
- 会话记忆
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json
import re


@dataclass
class Tool:
    """可调用的工具定义。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[[Dict[str, Any]], str]


@dataclass
class ConversationMemory:
    """会话记忆：保存原始对话和摘要。"""

    history: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    max_history: int = 8

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self._summarize_history()

    def _summarize_history(self) -> None:
        # 这里做最轻量的摘要：仅保留角色+内容的前 30 个字符
        summarized = []
        for item in self.history:
            snippet = item["content"][:30]
            summarized.append(f"{item['role']}:{snippet}")
        self.summary = " | ".join(summarized)
        self.history = self.history[-self.max_history :]

    def build_context(self) -> str:
        context_lines = []
        if self.summary:
            context_lines.append(f"摘要记忆：{self.summary}")
        for item in self.history:
            context_lines.append(f"{item['role']}：{item['content']}")
        return "\n".join(context_lines)


class SimpleAgent:
    """通过规则 + JSON 工具调用格式驱动的简易 Agent。"""

    def __init__(self, llm: Any, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}

    def _choose_tool_by_rule(self, task: str) -> Optional[str]:
        if "时间" in task or "几点" in task:
            return "get_time"
        if re.search(r"\d+\s*\+\s*\d+", task):
            return "add"
        if "检索" in task or "查资料" in task:
            return "rag_search"
        return None

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """允许模型输出 JSON 形式的工具调用。"""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        if "tool" not in data:
            return None
        return data

    def run(self, task: str, memory: ConversationMemory) -> str:
        memory.add_turn("user", task)
        tool_name = self._choose_tool_by_rule(task)
        if tool_name:
            tool = self.tools[tool_name]
            payload = self._extract_tool_payload(task, tool_name)
            tool_result = tool.func(payload)
            memory.add_turn("assistant", tool_result)
            return tool_result

        # 如果没有规则命中，则让模型决定是否要调用工具
        tool_instructions = [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.input_schema,
            }
            for tool in self.tools.values()
        ]
        prompt = (
            "你是一个工具调用助手。\n"
            "如果需要调用工具，请输出 JSON："
            "{\"tool\": \"工具名\", \"args\": {..}}。\n"
            "如果不需要调用工具，直接输出回答。\n"
            f"可用工具：{tool_instructions}\n"
            f"对话上下文：\n{memory.build_context()}\n"
            f"任务：{task}\n"
        )
        model_reply = self.llm.generate(prompt, max_output_tokens=512)
        tool_call = self._parse_tool_call(model_reply)
        if tool_call:
            tool = self.tools.get(tool_call["tool"])
            if tool:
                tool_result = tool.func(tool_call.get("args", {}))
                memory.add_turn("assistant", tool_result)
                return tool_result
        memory.add_turn("assistant", model_reply)
        return model_reply

    def _extract_tool_payload(self, task: str, tool_name: str) -> Dict[str, Any]:
        if tool_name == "add":
            numbers = re.findall(r"\d+(?:\.\d+)?", task)
            if len(numbers) >= 2:
                return {"a": float(numbers[0]), "b": float(numbers[1])}
        return {}
