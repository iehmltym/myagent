"""RAG（检索增强生成）核心组件的轻量实现。

包含：
- Document 数据结构
- Loader/切分器
- Embedding + 向量库
- Retriever + Rerank + 压缩
- 答案合成策略（stuff/map-reduce/refine）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re


@dataclass
class Document:
    """用于 RAG 的文档结构。"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleTextSplitter:
    """最基础的文本切分器：按字符长度切分并支持 overlap。"""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in docs:
            text = doc.content
            start = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                chunk_text = text[start:end]
                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata={**doc.metadata, "chunk_start": start, "chunk_end": end},
                    )
                )
                start = end - self.chunk_overlap
                if start < 0:
                    start = 0
        return chunks


class SimpleHashEmbedding:
    """用哈希将 token 映射到固定维度向量。避免外部依赖。"""

    def __init__(self, dims: int = 128):
        self.dims = dims

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def embed(self, text: str) -> List[float]:
        vector = [0.0 for _ in range(self.dims)]
        for token in self._tokenize(text):
            bucket = hash(token) % self.dims
            vector[bucket] += 1.0
        return vector


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class VectorStoreItem:
    doc: Document
    embedding: List[float]


class InMemoryVectorStore:
    """简易向量库：内存保存 embedding。"""

    def __init__(self, embedding_model: SimpleHashEmbedding):
        self.embedding_model = embedding_model
        self.items: List[VectorStoreItem] = []

    def add_documents(self, docs: Iterable[Document]) -> None:
        for doc in docs:
            embedding = self.embedding_model.embed(doc.content)
            self.items.append(VectorStoreItem(doc=doc, embedding=embedding))

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        query_vec = self.embedding_model.embed(query)
        results: List[Tuple[Document, float]] = []
        for item in self.items:
            if filters:
                if any(item.doc.metadata.get(k) != v for k, v in filters.items()):
                    continue
            score = _cosine_similarity(query_vec, item.embedding)
            if score >= score_threshold:
                results.append((item.doc, score))
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:top_k]


class VectorStoreRetriever:
    """检索器：支持基础相似度检索和简单 MMR。"""

    def __init__(
        self,
        store: InMemoryVectorStore,
        top_k: int = 4,
        score_threshold: float = 0.0,
        use_mmr: bool = False,
    ):
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_mmr = use_mmr

    def get_relevant_documents(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        candidates = self.store.similarity_search(
            query,
            top_k=max(self.top_k * 2, 8),
            score_threshold=self.score_threshold,
            filters=filters,
        )
        if not self.use_mmr:
            return candidates[: self.top_k]

        selected: List[Tuple[Document, float]] = []
        candidate_docs = [doc for doc, _ in candidates]
        candidate_scores = [score for _, score in candidates]
        while candidate_docs and len(selected) < self.top_k:
            best_index = 0
            best_score = -1.0
            for index, doc in enumerate(candidate_docs):
                diversity = 0.0
                for chosen_doc, _ in selected:
                    diversity += 1.0 - _cosine_similarity(
                        self.store.embedding_model.embed(doc.content),
                        self.store.embedding_model.embed(chosen_doc.content),
                    )
                combined = candidate_scores[index] + 0.3 * diversity
                if combined > best_score:
                    best_score = combined
                    best_index = index
            selected.append((candidate_docs.pop(best_index), candidate_scores.pop(best_index)))
        return selected


class OverlapReranker:
    """用 token overlap 的方式做轻量 rerank。"""

    def score(self, query: str, doc: Document) -> float:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        doc_tokens = set(re.findall(r"\w+", doc.content.lower()))
        if not query_tokens or not doc_tokens:
            return 0.0
        return len(query_tokens & doc_tokens) / len(query_tokens)

    def rerank(self, query: str, docs: List[Tuple[Document, float]], top_k: int = 4) -> List[Tuple[Document, float]]:
        scored = [(doc, self.score(query, doc)) for doc, _ in docs]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class ContextualCompressor:
    """根据 rerank 结果压缩上下文。"""

    def __init__(self, max_chars: int = 800):
        self.max_chars = max_chars

    def compress(self, docs: List[Tuple[Document, float]]) -> List[Document]:
        compressed: List[Document] = []
        for doc, score in docs:
            content = doc.content
            if len(content) > self.max_chars:
                content = content[: self.max_chars] + "..."
            compressed.append(Document(content=content, metadata={**doc.metadata, "score": score}))
        return compressed


class AnswerSynthesizer:
    """答案合成器：模拟 LangChain 的 stuff/map-reduce/refine。"""

    def __init__(self, llm: Any):
        self.llm = llm

    def synthesize(self, question: str, docs: List[Document], mode: str = "stuff") -> str:
        if not docs:
            return "未检索到相关资料。"
        if mode == "map_reduce":
            partials = []
            for doc in docs:
                prompt = f"根据以下资料回答问题：\n{doc.content}\n\n问题：{question}\n回答："
                partials.append(self.llm.generate(prompt, max_output_tokens=256))
            summary_prompt = "请汇总以下回答，给出最终答案：\n" + "\n".join(partials)
            return self.llm.generate(summary_prompt, max_output_tokens=256)
        if mode == "refine":
            answer = ""
            for doc in docs:
                prompt = (
                    "已有回答：{answer}\n"
                    "新资料：{doc}\n"
                    "问题：{question}\n"
                    "请在已有回答基础上补充与纠正："
                ).format(answer=answer, doc=doc.content, question=question)
                answer = self.llm.generate(prompt, max_output_tokens=256)
            return answer
        combined = "\n".join(doc.content for doc in docs)
        prompt = f"根据以下资料回答问题：\n{combined}\n\n问题：{question}\n回答："
        return self.llm.generate(prompt, max_output_tokens=256)


def evaluate_retrieval(retrieved: List[Tuple[Document, float]], expected_ids: List[str]) -> Dict[str, float]:
    """简单评估检索命中率。"""

    if not expected_ids:
        return {"hit_rate": 0.0}
    hit_count = 0
    for doc, _ in retrieved:
        doc_id = str(doc.metadata.get("id", ""))
        if doc_id in expected_ids:
            hit_count += 1
    return {"hit_rate": hit_count / len(expected_ids)}
