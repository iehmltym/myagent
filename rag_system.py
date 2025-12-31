from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleHashEmbedding:
    def __init__(self, dims: int = 128) -> None:
        self.dims = dims

    def embed(self, text: str) -> List[float]:
        values = [0.0] * self.dims
        for idx, ch in enumerate(text):
            values[idx % self.dims] += float(ord(ch) % 17)
        return values


class InMemoryVectorStore:
    def __init__(self, embedding_model: SimpleHashEmbedding) -> None:
        self.embedding_model = embedding_model
        self._docs: List[Document] = []

    def add_documents(self, documents: Iterable[Document]) -> None:
        self._docs.extend(list(documents))

    def similarity_search(self, query: str) -> List[Document]:
        return list(self._docs)


class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            text = doc.content
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunks.append(Document(content=chunk_text, metadata=dict(doc.metadata)))
                start = end - self.chunk_overlap
        return chunks


class VectorStoreRetriever:
    def __init__(self, store: InMemoryVectorStore, top_k: int = 4, score_threshold: float = 0.0, use_mmr: bool = False) -> None:
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_mmr = use_mmr

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.store.similarity_search(query)[: self.top_k]


class OverlapReranker:
    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        scored = []
        for doc in docs:
            score = sum(1 for token in query.split() if token in doc.content)
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class ContextualCompressor:
    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars

    def compress(self, docs: List[Document]) -> List[Document]:
        result = []
        total = 0
        for doc in docs:
            if total >= self.max_chars:
                break
            remaining = self.max_chars - total
            content = doc.content[:remaining]
            result.append(Document(content=content, metadata=dict(doc.metadata)))
            total += len(content)
        return result


class AnswerSynthesizer:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def synthesize(self, question: str, context: str) -> str:
        return f"问题：{question}\n\n参考资料：\n{context}"


def evaluate_retrieval(*_: Any, **__: Any) -> Dict[str, Any]:
    return {"status": "not_implemented"}
