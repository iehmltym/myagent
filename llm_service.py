from dataclasses import dataclass

from my_custom_llm import MyCustomGeminiLLM


@dataclass
class LLMService:
    cost_per_1k: float = 0.0

    def __post_init__(self) -> None:
        self.llm = MyCustomGeminiLLM(prefix="")
