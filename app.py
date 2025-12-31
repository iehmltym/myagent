from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import re
from typing import Dict, List, Optional, Tuple, Any

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
    # 不依赖 templates 文件，避免 Render 上找不到文件导致异常
    return """
    <!doctype html>
    <html lang="zh">
    <head><meta charset="utf-8"><title>MyAgent API</title></head>
    <body>
      <h2>MyAgent API is running</h2>
      <p>Health: <a href="/health">/health</a></p>
      <p>Use <code>POST /ask</code> with JSON: {"question":"..."}</p>
      <p>Use <code>POST /rag/ask</code> or <code>POST /agent/ask</code> for RAG/Agent demo.</p>
    </body>
    </html>
    """


@app.post("/ask")
def ask_question(req: QuestionRequest):
    # 取出用户输入的问题文本，便于后续处理
    question = req.question
    # 规则 1：如果问题中出现“现在几点”或“今天日期”，直接返回当前时间
    if "现在几点" in question or "今天日期" in question:
        def time_stream():
            yield get_time()

        return StreamingResponse(time_stream(), media_type="text/plain; charset=utf-8")

    # 规则 2：如果问题符合 “a+b=?” 形式，解析出 a、b 并计算
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


@app.post("/llm/complete")
def llm_complete(req: LLMRequest):
    """LLM 基础能力演示：同步输出 + token 估算 + 成本估算。"""
    span = trace_store.start_span("llm_complete", inputs={"prompt": req.prompt})
    completion, usage = llm_service.generate(
        req.prompt,
        max_output_tokens=req.max_output_tokens,
        temperature=req.temperature,
    )
    trace_store.finish_span(span, outputs={"completion": completion[:80]})
    return {
        "text": completion,
        "usage": usage.__dict__,
    }


@app.post("/llm/stream")
def llm_stream(req: LLMRequest):
    """LLM 流式输出演示。"""
    span = trace_store.start_span("llm_stream", inputs={"prompt": req.prompt})

    def stream():
        for chunk in llm_service.generate_stream(
            req.prompt, max_output_tokens=req.max_output_tokens, temperature=req.temperature
        ):
            yield chunk
        trace_store.finish_span(span, outputs={"completion": "streamed"})

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


@app.post("/llm/batch")
def llm_batch(req: BatchRequest):
    """LLM 批处理演示。"""
    span = trace_store.start_span("llm_batch", inputs={"size": str(len(req.prompts))})
    results = llm_service.generate_batch(req.prompts, max_output_tokens=req.max_output_tokens)
    trace_store.finish_span(span, outputs={"status": "ok"})
    return {
        "results": [
            {"text": text, "usage": usage.__dict__} for text, usage in results
        ]
    }


@app.post("/llm/async")
async def llm_async(req: LLMRequest):
    """LLM 异步调用演示。"""
    span = trace_store.start_span("llm_async", inputs={"prompt": req.prompt})
    completion, usage = await llm_service.generate_async(
        req.prompt,
        max_output_tokens=req.max_output_tokens,
        temperature=req.temperature,
    )
    trace_store.finish_span(span, outputs={"completion": completion[:80]})
    return {
        "text": completion,
        "usage": usage.__dict__,
    }


@app.post("/rag/ingest")
def rag_ingest(req: RAGIngestRequest):
    """文档接入：接收文档并执行切分 + 入库。"""
    span = trace_store.start_span("rag_ingest", inputs={"docs": str(len(req.documents))})
    docs = [Document(content=doc.content, metadata=doc.metadata) for doc in req.documents]
    chunks = text_splitter.split_documents(docs)
    vector_store.add_documents(chunks)
    trace_store.finish_span(span, outputs={"chunks": str(len(chunks))})
    return {"chunks": len(chunks)}


@app.post("/rag/ask")
def rag_ask(req: RAGAskRequest):
    """RAG 主流程：检索 -> rerank -> 压缩 -> 合成答案。"""
    span = trace_store.start_span("rag_ask", inputs={"question": req.question})
    # 检索阶段
    retrieved = retriever.get_relevant_documents(req.question)
    # rerank + 压缩上下文
    reranked = reranker.rerank(req.question, retrieved, top_k=req.top_k)
    compressed_docs = compressor.compress(reranked)
    # 合成答案
    answer = rag_synthesizer.synthesize(req.question, compressed_docs, mode=req.mode)
    # 简单评估
    metrics = evaluate_retrieval(retrieved, req.expected_ids or [])
    trace_store.finish_span(span, outputs={"answer": answer[:80]})
    return {
        "answer": answer,
        "references": [doc.metadata for doc in compressed_docs],
        "metrics": metrics,
    }


@app.post("/agent/ask")
def agent_ask(req: AgentRequest):
    """Agent 演示：根据任务调用工具或生成文本。"""
    session_id = req.session_id or "default"
    memory = agent_memories.setdefault(session_id, ConversationMemory())
    span = trace_store.start_span("agent_ask", inputs={"task": req.task})
    result = agent.run(req.task, memory)
    trace_store.finish_span(span, outputs={"result": result[:80]})
    return {
        "result": result,
        "memory": memory.build_context(),
    }


@app.get("/trace")
def trace_export():
    """导出最近的 trace 结果，便于调试。"""
    return {"spans": trace_store.export()}


@app.get("/lcel/demo")
def lcel_demo():
    """LCEL / Runnable 思路演示。"""
    template = PromptTemplate("请用一句话回答：{question}")
    chat_prompt = ChatPromptTemplate(
        [
            {"role": "system", "content": "你是一个严格的助教。"},
            {"role": "user", "content": "{question}"},
        ]
    )
    chain = RunnableSequence([template, LLMRunnable(llm_service.llm)])
    chat_chain = RunnableSequence([chat_prompt, LLMRunnable(llm_service.llm)])
    return {
        "prompt_chain": chain.invoke({"question": "什么是 RAG?"}),
        "chat_chain": chat_chain.invoke({"question": "什么是 Agent?"}),
    }
