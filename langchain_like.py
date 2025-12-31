from dataclasses import dataclass
from typing import Callable, List


@dataclass
class PromptTemplate:
    template: str

    def format(self, **kwargs: str) -> str:
        return self.template.format(**kwargs)


@dataclass
class ChatPromptTemplate:
    messages: List[str]

    def format(self, **kwargs: str) -> str:
        return "\n".join(self.messages).format(**kwargs)


@dataclass
class LLMRunnable:
    llm: Callable[[str], str]

    def invoke(self, prompt: str) -> str:
        return self.llm(prompt)


@dataclass
class RunnableSequence:
    runnables: List[Callable[[str], str]]

    def invoke(self, prompt: str) -> str:
        value = prompt
        for runnable in self.runnables:
            value = runnable(value)
        return value
