from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[[Dict[str, Any]], str]


@dataclass
class ConversationMemory:
    messages: List[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.messages.append(message)


class SimpleAgent:
    def __init__(self, llm: Any, tools: List[Tool]) -> None:
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}

    def run(self, task: str, tool_name: Optional[str] = None) -> str:
        if tool_name and tool_name in self.tools:
            return self.tools[tool_name].func({"query": task})
        return str(task)
