from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TraceStore:
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, trace: Dict[str, Any]) -> None:
        self.traces.append(trace)

    def list(self) -> List[Dict[str, Any]]:
        return list(self.traces)
