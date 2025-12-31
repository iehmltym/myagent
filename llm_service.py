"""LLM 调用相关能力：

- streaming / async / batch
- token 估算与成本计算
- 重试与退避
- 简单缓存与限流
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import asyncio
import time

from my_llm import MyGeminiLLM


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


def estimate_tokens(text: str) -> int:
    """用字符长度估算 token。

    经验值：1 token ≈ 4 个英文字符；中文按 1 字符估 1 token。
    这里只做演示，生产中应使用模型官方 tokenizer。
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class SimpleCache:
    """最小化缓存，支持 TTL。"""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._data: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._data.get(key)
        if not entry:
            return None
        created_at, value = entry
        if time.time() - created_at > self.ttl_seconds:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.time(), value)


class SimpleRateLimiter:
    """极简限流：确保请求间隔不低于 min_interval。"""

    def __init__(self, min_interval_seconds: float = 0.1):
        self.min_interval_seconds = min_interval_seconds
        self._last_time = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self._last_time
        if delta < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - delta)
        self._last_time = time.time()


class LLMService:
    """对 MyGeminiLLM 做更工程化的封装。"""

    def __init__(self, llm: Optional[MyGeminiLLM] = None, cost_per_1k: float = 0.0):
        self.llm = llm or MyGeminiLLM()
        self.cost_per_1k = cost_per_1k
        self.cache = SimpleCache()
        self.rate_limiter = SimpleRateLimiter()

    def _build_usage(self, prompt: str, completion: str) -> TokenUsage:
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(completion)
        total = prompt_tokens + completion_tokens
        cost = total / 1000 * self.cost_per_1k
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated_cost=cost,
        )

    def generate(self, prompt: str, max_retries: int = 2, **kwargs: Any) -> Tuple[str, TokenUsage]:
        cache_key = f"llm:{prompt}:{kwargs}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                self.rate_limiter.wait()
                completion = self.llm.generate(prompt, **kwargs)
                usage = self._build_usage(prompt, completion)
                result = (completion, usage)
                self.cache.set(cache_key, result)
                return result
            except Exception as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败: {last_error}")

    def generate_stream(self, prompt: str, **kwargs: Any):
        self.rate_limiter.wait()
        return self.llm.generate_stream(prompt, **kwargs)

    async def generate_async(self, prompt: str, **kwargs: Any) -> Tuple[str, TokenUsage]:
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    def generate_batch(self, prompts: Iterable[str], **kwargs: Any) -> List[Tuple[str, TokenUsage]]:
        results: List[Tuple[str, TokenUsage]] = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
