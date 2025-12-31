from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from my_custom_llm import MyCustomGeminiLLM
from tools import add, get_time
from llm_service import LLMService
from langchain_like import (
    ChatPromptTemplate,
    LLMRunnable,
    PromptTemplate,
    RunnableSequence,
)
from rag_system import (
    AnswerSynthesizer,
    ContextualCompressor,
    Document,
    InMemoryVectorStore,
    OverlapReranker,
    SimpleHashEmbedding,
    SimpleTextSplitter,
    VectorStoreRetriever,
    evaluate_retrieval,
)
from agent_system import ConversationMemory, SimpleAgent, Tool
from observability import TraceStore

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
# 会话历史字典：key 是 session_id，value 是 [(question, answer), ...]
# 用于在多轮对话中保留上下文，便于拼接成连续对话的 prompt
sessions: Dict[str, List[Tuple[str, str]]] = {}
# Agent 专用的记忆池：更适合工具调用的上下文管理
agent_memories: Dict[str, ConversationMemory] = {}

# 允许你的 GitHub Pages 调用（/ai 仍属于同一域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://iehmltym.github.io"],
    allow_origin_regex=r"https://iehmltym\.github\.io$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# 全局默认的 LLM 服务封装（包含缓存、限流、重试等工程能力）
llm_service = LLMService(cost_per_1k=0.0)
# 默认的可爱语气 system prompt（当用户未提供时使用）
DEFAULT_CUTE_SYSTEM_PROMPT = "请用可爱的语气回答，简洁、温柔，像小可爱一样～"

# ===== RAG 的全局组件 =====
embedding_model = SimpleHashEmbedding(dims=128)
vector_store = InMemoryVectorStore(embedding_model=embedding_model)
text_splitter = SimpleTextSplitter(chunk_size=400, chunk_overlap=80)
retriever = VectorStoreRetriever(store=vector_store, top_k=4, score_threshold=0.1, use_mmr=True)
reranker = OverlapReranker()
compressor = ContextualCompressor(max_chars=800)
rag_synthesizer = AnswerSynthesizer(llm=llm_service.llm)

# 默认示例知识库：让 RAG 即开即用（可通过 /rag/ingest 覆盖/追加）
default_docs = [
    Document(
        content="RAG 是检索增强生成，通过向量检索获取外部资料，再让模型综合回答。",
        metadata={"id": "rag-001", "source": "builtin"},
    ),
    Document(
        content="Agent 适合开放式任务编排，可结合工具调用与记忆管理完成复杂任务。",
        metadata={"id": "agent-001", "source": "builtin"},
    ),
    Document(
        content="LCEL/Runnable 思维强调可组合性：prompt -> model -> parser -> tool。",
        metadata={"id": "lcel-001", "source": "builtin"},
    ),
]
vector_store.add_documents(text_splitter.split_documents(default_docs))

# ===== 观测能力 =====
trace_store = TraceStore()

# ===== Agent 工具配置 =====
def tool_add(args: Dict[str, Any]) -> str:
    """工具：执行加法。"""
    return str(add(float(args.get("a", 0)), float(args.get("b", 0))))


def tool_time(_: Dict[str, Any]) -> str:
    """工具：获取当前时间。"""
    return get_time()


def tool_rag_search(args: Dict[str, Any]) -> str:
    """工具：执行 RAG 检索并返回摘要。"""
    query = str(args.get("query", ""))
    docs = retriever.get_relevant_documents(query)
    reranked = reranker.rerank(query, docs, top_k=3)
    compressed = compressor.compress(reranked)
    return "\n".join(doc.content for doc in compressed)


tools = [
    Tool(
        name="add",
        description="计算两个数字的和",
        input_schema={"a": "number", "b": "number"},
        func=tool_add,
    ),
    Tool(
        name="get_time",
        description="获取当前时间",
        input_schema={},
        func=tool_time,
    ),
    Tool(
        name="rag_search",
        description="检索本地知识库并返回相关片段",
        input_schema={"query": "string"},
        func=tool_rag_search,
    ),
]
agent = SimpleAgent(llm=llm_service.llm, tools=tools)


class QuestionRequest(BaseModel):
    # 用户输入的问题文本
    question: str
    # 可选的 system prompt，用来影响模型输出风格
    # 如果不传或传空字符串，就会走默认的 MyGeminiLLM
    system_prompt: Optional[str] = None
    # 会话 ID：用于区分不同用户/对话的上下文
    # 为空时表示单轮请求，不记录历史
    session_id: Optional[str] = None


class LLMRequest(BaseModel):
    # 模型输入
    prompt: str
    # 控制输出长度
    max_output_tokens: int = 512
    # 温度参数
    temperature: float = 0.7


class BatchRequest(BaseModel):
    # 多条 prompt 输入
    prompts: List[str]
    max_output_tokens: int = 256


class IngestDocument(BaseModel):
    # 文档正文
    content: str
    # 元数据（可选）
    metadata: Dict[str, Any] = {}


class RAGIngestRequest(BaseModel):
    # 可一次性传入多个文档
    documents: List[IngestDocument]


class RAGAskRequest(BaseModel):
    question: str
    mode: str = "stuff"
    top_k: int = 4
    expected_ids: Optional[List[str]] = None


class AgentRequest(BaseModel):
    task: str
    session_id: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    return (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/entry", response_class=HTMLResponse)
def entry():
    return (TEMPLATE_DIR / "entry.html").read_text(encoding="utf-8")


@app.post("/ask")
def ask_question(req: QuestionRequest):
    # 取出用户输入的问题文本，便于后续处理
    question = req.question
    # 规则 1：如果问题中出现“现在几点”或“今天日期”，直接返回当前时间
    if "现在几点" in question or "今天日期" in question:
        def time_stream():
            yield get_time()

        return StreamingResponse(time_stream(), media_type="text/plain; charset=utf-8")

    # 规则 2：如果问题符合 “a+b=?" 形式，解析出 a、b 并计算
    # 说明：下面这个正则允许空格和小数，比如 " 1 + 2 = ? "
    math_match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*=\s*\?\s*", question)
    if math_match:
        # 正则分组 1 和 2 分别是 a、b 的文本形式
        left = float(math_match.group(1))
        right = float(math_match.group(2))
        # 计算完成后直接返回，避免走 LLM
        def math_stream():
            yield str(add(left, right))

        return StreamingResponse(math_stream(), media_type="text/plain; charset=utf-8")

    # 如果前端传了 system_prompt，就用它；
    # 否则使用默认的可爱语气 prompt。
    system_prompt = req.system_prompt or DEFAULT_CUTE_SYSTEM_PROMPT
    # 始终用自定义 LLM 包装器注入 system prompt，让回答保持可爱风格
    active_llm = MyCustomGeminiLLM(prefix=system_prompt)
    # 规范化 session_id（去掉首尾空白），避免同一会话被当成多个 key
    session_id = req.session_id.strip() if req.session_id else None
    # 获取该会话的历史；若不存在则初始化为空列表
    # 未提供 session_id 时视为单轮对话，不读取/写入历史
    history = sessions.setdefault(session_id, []) if session_id else []
    # 将历史记录拼成连续对话的 prompt，格式如：
    # 用户：... \n 助手：... \n 用户：... \n 助手：...
    history_prompt = "".join(
        f"用户：{question}\n助手：{response}\n" for question, response in history
    )
    # 拼接本次问题，提示模型继续回复助手内容
    prompt = f"{history_prompt}用户：{req.question}\n助手："
    # 调用模型生成答案，max_output_tokens 适当提高以避免回答被截断
    def answer_stream():
        answer_parts = []
        for chunk in active_llm.generate_stream(prompt, max_output_tokens=2048):
            answer_parts.append(chunk)
            yield chunk
        if session_id:
            answer = "".join(answer_parts)
            # 仅当 session_id 有效时才记录历史，避免无意义的全局堆积
            history.append((req.question, answer))

    return StreamingResponse(answer_stream(), media_type="text/plain; charset=utf-8")
