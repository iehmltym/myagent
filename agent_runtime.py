from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from my_llm import MyGeminiLLM


@dataclass
class PromptMessage:
    role: str
    content: str


@dataclass
class PromptTemplate:
    system: str
    messages: List[PromptMessage] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        system_text = self._format(self.system)
        rendered_messages = [self._format(msg.content) for msg in self.messages]
        parts = [system_text] if system_text else []
        for message, content in zip(self.messages, rendered_messages):
            parts.append(f"{message.role}：{content}")
        return "\n".join(parts).strip()

    def _format(self, text: str) -> str:
        if not text:
            return ""
        try:
            return text.format(**self.variables)
        except KeyError:
            return text


@dataclass
class ToolResult:
    output: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[[str], ToolResult]
    keywords: Sequence[str] = field(default_factory=list)

    def match(self, query: str) -> bool:
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in self.keywords)


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        self._tools: List[Tool] = list(tools or [])

    def register(self, tool: Tool) -> None:
        self._tools.append(tool)

    def match(self, query: str) -> Optional[Tool]:
        for tool in self._tools:
            if tool.match(query):
                return tool
        return None

    def get(self, name: str) -> Optional[Tool]:
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def summary(self) -> List[Dict[str, str]]:
        return [{"name": tool.name, "desc": tool.description} for tool in self._tools]


class MemoryStore:
    def __init__(self, max_turns: int, max_sessions: int):
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self.sessions: "Dict[str, List[Tuple[str, str]]]" = {}
        self._order: List[str] = []

    def get(self, session_id: Optional[str]) -> List[Tuple[str, str]]:
        if not session_id:
            return []
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            self._order.append(session_id)
        self._touch(session_id)
        self._trim_sessions()
        return self.sessions[session_id]

    def append(self, session_id: Optional[str], question: str, answer: str) -> None:
        if not session_id:
            return
        history = self.get(session_id)
        history.append((question, answer))
        if len(history) > self.max_turns:
            self.sessions[session_id] = history[-self.max_turns :]
        self._touch(session_id)

    def clear(self, session_id: str) -> bool:
        if session_id in self.sessions:
            self.sessions.pop(session_id, None)
            if session_id in self._order:
                self._order.remove(session_id)
            return True
        return False

    def active_count(self) -> int:
        return len(self.sessions)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        session_ids = list(reversed(self._order))[:limit]
        return [
            {
                "session_id": session_id,
                "turns": len(self.sessions.get(session_id, [])),
            }
            for session_id in session_ids
        ]

    def export(self, session_id: str) -> List[Tuple[str, str]]:
        return list(self.sessions.get(session_id, []))

    def _touch(self, session_id: str) -> None:
        if session_id in self._order:
            self._order.remove(session_id)
        self._order.append(session_id)

    def _trim_sessions(self) -> None:
        while len(self._order) > self.max_sessions:
            oldest = self._order.pop(0)
            self.sessions.pop(oldest, None)


@dataclass
class LLMRequestConfig:
    temperature: float = 0.7
    max_output_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 1


class LLMProvider:
    name: str = "base"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, config: LLMRequestConfig) -> str:
        raise NotImplementedError

    def generate_stream(self, prompt: str, config: LLMRequestConfig):
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self):
        self._client: Optional[MyGeminiLLM] = None
        self._init_error: Optional[str] = None
        try:
            self._client = MyGeminiLLM()
        except Exception as exc:  # noqa: BLE001 - capture init failure
            self._init_error = str(exc)

    def is_available(self) -> bool:
        return self._client is not None

    def _unavailable_message(self) -> str:
        hint = self._init_error or "Gemini SDK 未就绪。"
        return f"[gemini] 不可用：{hint}"

    def generate(self, prompt: str, config: LLMRequestConfig) -> str:
        if not self._client:
            return self._unavailable_message()
        return self._client.generate(
            prompt,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
        )

    def generate_stream(self, prompt: str, config: LLMRequestConfig):
        if not self._client:
            yield self._unavailable_message()
            return
        return self._client.generate_stream(
            prompt,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
        )


class PlaceholderProvider(LLMProvider):
    def __init__(self, name: str, hint: str):
        self.name = name
        self._hint = hint

    def is_available(self) -> bool:
        return False

    def generate(self, prompt: str, config: LLMRequestConfig) -> str:
        return f"[{self.name}] 尚未配置：{self._hint}"

    def generate_stream(self, prompt: str, config: LLMRequestConfig):
        yield self.generate(prompt, config)


class LLMManager:
    def __init__(self, providers: Iterable[LLMProvider], default_provider: str = "gemini"):
        self.providers = {provider.name: provider for provider in providers}
        self.default_provider = default_provider

    def route(self, question: str, provider: Optional[str], enable_router: bool) -> Tuple[str, str]:
        if provider and provider != "auto":
            return provider, "manual"
        if not enable_router:
            return self.default_provider, "router_off"
        if len(question) > 120 or any(keyword in question for keyword in ("架构", "方案", "升级")):
            return self.default_provider, "complex"
        return self.default_provider, "default"

    def generate(
        self,
        prompt: str,
        config: LLMRequestConfig,
        question: str,
        provider: Optional[str],
        fallback: Sequence[str],
        retries: int,
        enable_router: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        route_provider, route_reason = self.route(question, provider, enable_router)
        attempts = [route_provider, *fallback]
        last_error: Optional[Exception] = None
        for attempt in attempts:
            active = self.providers.get(attempt)
            if not active:
                continue
            if not active.is_available() and attempt != self.default_provider:
                continue
            for _ in range(max(retries, 1)):
                try:
                    return (
                        active.generate(prompt, config),
                        {"provider": attempt, "route_reason": route_reason},
                    )
                except Exception as exc:  # noqa: BLE001 - surface error in meta
                    last_error = exc
                    continue
        fallback_provider = self.providers.get(self.default_provider)
        if fallback_provider:
            return (
                fallback_provider.generate(prompt, config),
                {"provider": self.default_provider, "route_reason": "fallback"},
            )
        if last_error:
            raise last_error
        raise RuntimeError("没有可用的 LLM Provider")

    def generate_stream(
        self,
        prompt: str,
        config: LLMRequestConfig,
        question: str,
        provider: Optional[str],
        fallback: Sequence[str],
        retries: int,
        enable_router: bool,
    ) -> Tuple[Iterable[str], Dict[str, Any]]:
        route_provider, route_reason = self.route(question, provider, enable_router)
        attempts = [route_provider, *fallback]
        for attempt in attempts:
            active = self.providers.get(attempt)
            if not active:
                continue
            if not active.is_available() and attempt != self.default_provider:
                continue
            for _ in range(max(retries, 1)):
                try:
                    return (
                        active.generate_stream(prompt, config),
                        {"provider": attempt, "route_reason": route_reason},
                    )
                except Exception:
                    continue
        fallback_provider = self.providers.get(self.default_provider)
        if fallback_provider:
            return (
                fallback_provider.generate_stream(prompt, config),
                {"provider": self.default_provider, "route_reason": "fallback"},
            )
        raise RuntimeError("没有可用的 LLM Provider")


class Chain:
    def __init__(self, steps: Sequence[Callable[[Dict[str, Any]], Dict[str, Any]]]):
        self.steps = steps

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(payload)
        for step in self.steps:
            state = step(state)
        return state
