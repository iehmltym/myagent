from collections import OrderedDict, deque
from datetime import datetime
import re
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent_runtime import (
    Chain,
    GeminiProvider,
    LLMManager,
    LLMRequestConfig,
    MemoryStore,
    PlaceholderProvider,
    PromptMessage,
    PromptTemplate,
    Tool,
    ToolRegistry,
    ToolResult,
)
from knowledge_base import blueprint_payload, retrieve_documents
from tools import add, get_date, get_time, make_uuid, safe_eval_expression, text_stats

app = FastAPI()
MAX_HISTORY_TURNS = 10
MAX_SESSIONS = 200
STARTED_AT = time.monotonic()
STARTED_AT_WALL = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
OBS_LOG_LIMIT = 200

answer_cache: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
MAX_CACHE_ITEMS = 256
cache_stats = {"hits": 0, "misses": 0}
request_stats = {"total": 0, "llm": 0, "rule": 0, "cache": 0}
observability_log: Deque[Dict[str, Any]] = deque(maxlen=OBS_LOG_LIMIT)

COMMAND_PATTERN = re.compile(r"^/([a-zA-Z]+)\s*(.*)$")
MATH_PATTERN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*=\s*\?\s*")

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

# 默认的可爱语气 system prompt（当用户未提供时使用）
DEFAULT_CUTE_SYSTEM_PROMPT = "你是林志玲今日值班AI小助手，请用可爱的语气回答，简洁、温柔，像小可爱一样～"
ADVISOR_SYSTEM_PROMPT = (
    "你是资深 AI 架构顾问，擅长 LangChain/Agent/RAG 项目落地。"
    "请用中文回答，结构必须包含："
    "1) 核心能力（LangChain 能做什么）"
    "2) 很强的典型用法（落到真实项目）"
    "3) Render 512MB 免费实例的落地建议"
    "4) 项目下一步高价值升级"
    "给出清晰的小标题和要点列表，语气专业、直接。"
)

memory_store = MemoryStore(max_turns=MAX_HISTORY_TURNS, max_sessions=MAX_SESSIONS)
tool_registry = ToolRegistry(
    tools=[
        Tool(
            name="get_time",
            description="获取当前时间",
            handler=lambda _: ToolResult(get_time(), {"tool": "time"}),
            keywords=["现在几点", "时间", "time"],
        ),
        Tool(
            name="get_date",
            description="获取今天日期",
            handler=lambda _: ToolResult(get_date(), {"tool": "date"}),
            keywords=["今天日期", "日期", "date"],
        ),
        Tool(
            name="uuid",
            description="生成 UUID4",
            handler=lambda _: ToolResult(make_uuid(), {"tool": "uuid"}),
            keywords=["uuid", "id"],
        ),
        Tool(
            name="calc",
            description="安全计算数学表达式",
            handler=lambda text: ToolResult(str(safe_eval_expression(text)), {"tool": "calc"}),
            keywords=["计算", "算一下", "calc"],
        ),
        Tool(
            name="text_stats",
            description="统计文本长度与单词数",
            handler=lambda text: ToolResult(
                (
                    f"字符数：{text_stats(text)['chars']} | "
                    f"去空格字符数：{text_stats(text)['chars_no_space']} | "
                    f"单词数：{text_stats(text)['words']}"
                ),
                {"tool": "len"},
            ),
            keywords=["字数", "文本统计", "len"],
        ),
    ]
)
llm_manager = LLMManager(
    providers=[
        GeminiProvider(),
        PlaceholderProvider("openai", "请配置 OPENAI_API_KEY 并启用对应 SDK。"),
        PlaceholderProvider("anthropic", "请配置 ANTHROPIC_API_KEY 并启用对应 SDK。"),
        PlaceholderProvider("ollama", "请启动本地 Ollama 服务并配置地址。"),
    ]
)


class QuestionRequest(BaseModel):
    # 用户输入的问题文本
    question: str
    # 可选的 system prompt，用来影响模型输出风格
    # 如果不传或传空字符串，就会走默认的 MyGeminiLLM
    system_prompt: Optional[str] = None
    # 会话 ID：用于区分不同用户/对话的上下文
    # 为空时表示单轮请求，不记录历史
    session_id: Optional[str] = None
    # 是否启用流式响应（默认 True 兼容旧版前端）
    stream: bool = True
    # 是否返回元信息（如缓存命中、耗时等）
    include_meta: bool = False
    # 运行模式：默认空；"advisor" 表示架构建议；"rag" 表示检索增强
    mode: Optional[str] = None
    # LLM Provider：auto/gemini/openai/anthropic/ollama
    provider: Optional[str] = None
    # LLM 参数
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    # 工具调用与路由开关
    enable_tools: bool = True
    enable_router: bool = True
    # 回退与重试
    fallback_providers: Optional[List[str]] = None
    retries: int = 1


class SessionRequest(BaseModel):
    session_id: str


class ToolRunRequest(BaseModel):
    name: str
    payload: Optional[str] = ""


class PromptPreviewRequest(BaseModel):
    question: str
    system_prompt: Optional[str] = None
    mode: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = 4
    provider: Optional[str] = None
    enable_router: bool = True


class RAGPreviewRequest(BaseModel):
    query: str
    top_k: int = 4


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    template_path = "templates/index.html"
    try:
        return FileResponse(template_path)
    except FileNotFoundError:
        return """
        <!doctype html>
        <html lang="zh">
        <head><meta charset="utf-8"><title>MyAgent API</title></head>
        <body>
          <h2>MyAgent API is running</h2>
          <p>Health: <a href="/health">/health</a></p>
          <p>Use <code>POST /ask</code> with JSON: {"question":"..."}</p>
        </body>
        </html>
        """


@app.get("/features")
def features() -> Dict[str, Any]:
    return {
        "commands": [
            {"cmd": "/time", "desc": "查看当前时间"},
            {"cmd": "/date", "desc": "查看今日日期"},
            {"cmd": "/uuid", "desc": "生成 UUID4"},
            {"cmd": "/calc 1+2*3", "desc": "安全计算表达式"},
            {"cmd": "/len 文本", "desc": "查看文本统计"},
            {"cmd": "/help", "desc": "查看可用命令"},
        ],
        "providers": [
            {"id": "auto", "desc": "自动路由（建议默认）"},
            {"id": "gemini", "desc": "Google Gemini（已接入）"},
            {"id": "openai", "desc": "OpenAI（占位，需配置）"},
            {"id": "anthropic", "desc": "Anthropic（占位，需配置）"},
            {"id": "ollama", "desc": "本地 Ollama（占位，需配置）"},
        ],
        "prompt": {
            "supports": ["system", "messages", "few-shot", "variables", "versioning"],
            "template": "PromptTemplate + Messages",
        },
        "chains": {
            "steps": ["preprocess", "tool", "retrieve", "generate", "postprocess"],
            "desc": "用户输入 → 预处理 → 检索 → 生成 → 后处理",
        },
        "tools": tool_registry.summary(),
        "memory": {
            "window_turns": MAX_HISTORY_TURNS,
            "long_term": "vector-store (规划中，512MB 建议外部托管)",
        },
        "limits": {
            "max_history_turns": MAX_HISTORY_TURNS,
            "max_sessions": MAX_SESSIONS,
            "cache_size": MAX_CACHE_ITEMS,
        },
        "modes": [
            {"id": "advisor", "desc": "架构建议模式：按固定结构输出升级建议"},
            {"id": "rag", "desc": "检索增强模式：结合内置资料输出"},
        ],
        "admin": {
            "endpoints": [
                {"path": "/system", "desc": "服务运行状态与限制信息"},
                {"path": "/cache/clear", "desc": "清空回答缓存"},
                {"path": "/session/list", "desc": "查看活跃会话列表"},
                {"path": "/session/export", "desc": "导出会话历史"},
                {"path": "/observability/logs", "desc": "查看最近请求日志"},
                {"path": "/observability/clear", "desc": "清空观测日志"},
                {"path": "/router/explain", "desc": "Router 决策解释"},
                {"path": "/tools/run", "desc": "直接调用指定工具"},
                {"path": "/rag/preview", "desc": "预览 RAG 检索结果"},
                {"path": "/prompt/preview", "desc": "预览最终 Prompt"},
            ]
        },
    }


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return {
        "requests": request_stats,
        "cache": {"size": len(answer_cache), **cache_stats},
        "sessions": {"active": memory_store.active_count(), "max": MAX_SESSIONS},
    }


@app.get("/observability/logs")
def observability_logs(limit: int = 50) -> Dict[str, Any]:
    safe_limit = max(1, min(limit, OBS_LOG_LIMIT))
    entries = list(observability_log)[-safe_limit:]
    return {"events": entries, "count": len(entries), "limit": OBS_LOG_LIMIT}


@app.post("/observability/clear")
def observability_clear() -> Dict[str, Any]:
    cleared = len(observability_log)
    observability_log.clear()
    return {"cleared": cleared, "remaining": len(observability_log)}


@app.get("/system")
def system_info() -> Dict[str, Any]:
    uptime_seconds = round(time.monotonic() - STARTED_AT, 2)
    return {
        "started_at": STARTED_AT_WALL,
        "uptime_seconds": uptime_seconds,
        "limits": {
            "max_history_turns": MAX_HISTORY_TURNS,
            "max_sessions": MAX_SESSIONS,
            "cache_size": MAX_CACHE_ITEMS,
        },
        "cache": {"size": len(answer_cache), **cache_stats},
        "sessions": {"active": memory_store.active_count(), "max": MAX_SESSIONS},
        "observability": {"events": len(observability_log), "limit": OBS_LOG_LIMIT},
    }


@app.get("/blueprint")
def blueprint() -> Dict[str, Any]:
    return blueprint_payload()


@app.post("/session/clear")
def clear_session(req: SessionRequest) -> Dict[str, Any]:
    if memory_store.clear(req.session_id):
        return {"cleared": True, "session_id": req.session_id}
    return {"cleared": False, "session_id": req.session_id}


@app.get("/session/list")
def list_sessions(limit: int = 50) -> Dict[str, Any]:
    safe_limit = max(1, min(limit, MAX_SESSIONS))
    return {"sessions": memory_store.list_sessions(safe_limit)}


@app.post("/session/export")
def export_session(req: SessionRequest) -> Dict[str, Any]:
    history = memory_store.export(req.session_id)
    return {
        "session_id": req.session_id,
        "history": history,
        "missing": not bool(history),
    }


@app.post("/cache/clear")
def clear_cache() -> Dict[str, Any]:
    cleared = len(answer_cache)
    answer_cache.clear()
    cache_stats["hits"] = 0
    cache_stats["misses"] = 0
    return {"cleared": cleared, "cache_size": len(answer_cache)}


@app.get("/router/explain")
def router_explain(question: str, provider: Optional[str] = None, enable_router: bool = True) -> Dict[str, Any]:
    chosen, reason = llm_manager.route(question, provider, enable_router)
    return {"provider": chosen, "reason": reason}


@app.post("/tools/run")
def run_tool(req: ToolRunRequest) -> Dict[str, Any]:
    tool = tool_registry.get(req.name.strip())
    if not tool:
        return {"ok": False, "error": f"未知工具：{req.name}"}
    try:
        result = tool.handler(req.payload or "")
    except ValueError as exc:
        return {"ok": False, "error": f"工具调用失败：{exc}"}
    return {"ok": True, "output": result.output, "meta": result.meta}


@app.post("/rag/preview")
def rag_preview(req: RAGPreviewRequest) -> Dict[str, Any]:
    docs = retrieve_documents(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "count": len(docs),
        "documents": docs,
    }


@app.post("/prompt/preview")
def prompt_preview(req: PromptPreviewRequest) -> Dict[str, Any]:
    system_prompt = req.system_prompt or DEFAULT_CUTE_SYSTEM_PROMPT
    mode = (req.mode or "").strip().lower()
    rag_context = ""
    rag_meta: Dict[str, Any] = {}
    if mode in {"advisor", "rag"}:
        if not req.system_prompt:
            system_prompt = ADVISOR_SYSTEM_PROMPT
        rag_docs = retrieve_documents(req.question, top_k=req.top_k)
        if rag_docs:
            rag_context = "\n\n".join(
                f"[{index + 1}] {doc['title']}\n{doc['content']}" for index, doc in enumerate(rag_docs)
            )
            rag_meta = {
                "rag_docs": [doc["title"] for doc in rag_docs],
                "rag_count": len(rag_docs),
            }

    session_id = req.session_id.strip() if req.session_id else None
    history = memory_store.get(session_id) if session_id else []
    history_prompt = "".join(f"用户：{user}\n助手：{reply}\n" for user, reply in history)

    messages = []
    if history_prompt:
        messages.append(PromptMessage(role="用户", content=history_prompt))
    messages.append(PromptMessage(role="用户", content="{question}"))
    prompt_template = PromptTemplate(
        system=system_prompt,
        messages=messages,
        variables={"question": req.question},
    )
    if rag_context:
        prompt = (
            f"{prompt_template.render()}\n\n"
            f"请参考以下资料作答（不要逐字复述，保持结构化输出）：\n{rag_context}\n\n"
            "助手："
        )
    else:
        prompt = f"{prompt_template.render()}\n助手："
    provider, route_reason = llm_manager.route(req.question, req.provider, req.enable_router)
    return {
        "prompt": prompt,
        "mode": mode or "default",
        "provider": provider,
        "route_reason": route_reason,
        **rag_meta,
    }

def _cache_get(key: Tuple[str, str]) -> Optional[str]:
    if key in answer_cache:
        answer_cache.move_to_end(key)
        cache_stats["hits"] += 1
        request_stats["cache"] += 1
        return answer_cache[key]
    cache_stats["misses"] += 1
    return None


def _cache_set(key: Tuple[str, str], answer: str) -> None:
    answer_cache[key] = answer
    answer_cache.move_to_end(key)
    if len(answer_cache) > MAX_CACHE_ITEMS:
        answer_cache.popitem(last=False)


def _handle_command(question: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    match = COMMAND_PATTERN.match(question.strip())
    if not match:
        return None
    cmd = match.group(1).lower()
    payload = match.group(2).strip()
    if cmd == "time":
        return get_time(), {"command": "/time"}
    if cmd == "date":
        return get_date(), {"command": "/date"}
    if cmd == "uuid":
        return make_uuid(), {"command": "/uuid"}
    if cmd == "calc":
        if not payload:
            return "请输入表达式，例如 /calc 1+2*3", {"command": "/calc"}
        try:
            return str(safe_eval_expression(payload)), {"command": "/calc"}
        except ValueError as exc:
            return f"计算失败：{exc}", {"command": "/calc", "error": True}
    if cmd == "len":
        if not payload:
            return "请输入要统计的文本，例如 /len hello world", {"command": "/len"}
        stats = text_stats(payload)
        return (
            f"字符数：{stats['chars']} | 去空格字符数：{stats['chars_no_space']} | 单词数：{stats['words']}",
            {"command": "/len"},
        )
    if cmd == "help":
        return (
            "可用命令：/time /date /uuid /calc 表达式 /len 文本 /help",
            {"command": "/help"},
        )
    return "未知命令，试试 /help", {"command": f"/{cmd}", "error": True}


def _auto_tool(question: str) -> Optional[ToolResult]:
    tool = tool_registry.match(question)
    if not tool:
        return None
    payload = question
    if tool.name in {"calc", "text_stats"}:
        payload = question.replace("计算", "").replace("算一下", "").replace("字数", "").strip()
    if tool.name == "calc" and not payload:
        payload = question
    try:
        return tool.handler(payload)
    except ValueError as exc:
        return ToolResult(f"工具调用失败：{exc}", {"tool": tool.name, "error": True})


@app.post("/ask")
def ask_question(req: QuestionRequest):
    # 取出用户输入的问题文本，便于后续处理
    question = req.question.strip()
    request_id = str(uuid.uuid4())
    stream = req.stream
    include_meta = req.include_meta
    forced_stream_off = False
    if include_meta and stream:
        stream = False
        forced_stream_off = True

    started = time.monotonic()
    request_stats["total"] += 1

    def respond_text(answer: str, *, meta: Optional[Dict[str, Any]] = None):
        headers = {"X-Request-ID": request_id}
        if stream:
            def answer_stream():
                yield answer

            if meta:
                headers["X-Answer-Source"] = meta.get("source", "")
                headers["X-Cache-Hit"] = str(meta.get("cache_hit", False)).lower()
            return StreamingResponse(answer_stream(), media_type="text/plain; charset=utf-8", headers=headers)
        payload = {"answer": answer}
        if include_meta and meta:
            payload["meta"] = meta
        return JSONResponse(payload, headers=headers)

    def update_history(session_key: Optional[str], answer: str):
        memory_store.append(session_key, question, answer)

    def build_meta(source: str, cache_hit: bool, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = {
            "source": source,
            "cache_hit": cache_hit,
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
            "request_id": request_id,
        }
        if forced_stream_off:
            meta["stream_forced_off"] = True
        if extra:
            meta.update(extra)
        return meta

    def log_event(meta: Dict[str, Any]) -> None:
        observability_log.append(
            {
                "request_id": meta.get("request_id"),
                "source": meta.get("source"),
                "latency_ms": meta.get("latency_ms"),
                "cache_hit": meta.get("cache_hit"),
                "provider": meta.get("provider"),
                "route_reason": meta.get("route_reason"),
                "question": question,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    command_result = _handle_command(question)
    if command_result:
        answer, extra = command_result
        request_stats["rule"] += 1
        meta = build_meta("command", False, extra)
        log_event(meta)
        return respond_text(answer, meta=meta)
    # 规则 1：如果问题中出现“现在几点”或“今天日期”，直接返回当前时间
    if "现在几点" in question or "今天日期" in question:
        request_stats["rule"] += 1
        meta = build_meta("rule", False, {"tool": "time"})
        log_event(meta)
        return respond_text(get_time(), meta=meta)

    # 规则 2：如果问题符合 “a+b=?” 形式，解析出 a、b 并计算
    # 说明：下面这个正则允许空格和小数，比如 " 1 + 2 = ? "
    math_match = MATH_PATTERN.fullmatch(question)
    if math_match:
        # 正则分组 1 和 2 分别是 a、b 的文本形式
        left = float(math_match.group(1))
        right = float(math_match.group(2))
        # 计算完成后直接返回，避免走 LLM
        request_stats["rule"] += 1
        meta = build_meta("rule", False, {"tool": "add"})
        log_event(meta)
        return respond_text(str(add(left, right)), meta=meta)

    if req.enable_tools:
        tool_result = _auto_tool(question)
        if tool_result:
            request_stats["rule"] += 1
            meta = build_meta("tool", False, tool_result.meta)
            log_event(meta)
            return respond_text(tool_result.output, meta=meta)

    # 如果前端传了 system_prompt，就用它；
    # 否则使用默认的可爱语气 prompt。
    system_prompt = req.system_prompt or DEFAULT_CUTE_SYSTEM_PROMPT
    mode = (req.mode or "").strip().lower()
    rag_context = ""
    rag_meta: Dict[str, Any] = {}
    if mode in {"advisor", "rag"}:
        if not req.system_prompt:
            system_prompt = ADVISOR_SYSTEM_PROMPT
        rag_docs = retrieve_documents(question, top_k=4)
        if rag_docs:
            rag_context = "\n\n".join(
                f"[{index + 1}] {doc['title']}\n{doc['content']}" for index, doc in enumerate(rag_docs)
            )
            rag_meta = {
                "rag_docs": [doc["title"] for doc in rag_docs],
                "rag_count": len(rag_docs),
            }
    # 规范化 session_id（去掉首尾空白），避免同一会话被当成多个 key
    session_id = req.session_id.strip() if req.session_id else None
    # 获取该会话的历史；若不存在则初始化为空列表
    # 未提供 session_id 时视为单轮对话，不读取/写入历史
    history = memory_store.get(session_id) if session_id else []
    history_prompt = "".join(f"用户：{user}\n助手：{reply}\n" for user, reply in history)

    messages = []
    if history_prompt:
        messages.append(PromptMessage(role="用户", content=history_prompt))
    messages.append(PromptMessage(role="用户", content="{question}"))
    prompt_template = PromptTemplate(
        system=system_prompt,
        messages=messages,
        variables={"question": req.question},
    )

    if rag_context:
        prompt = (
            f"{prompt_template.render()}\n\n"
            f"请参考以下资料作答（不要逐字复述，保持结构化输出）：\n{rag_context}\n\n"
            "助手："
        )
    else:
        prompt = f"{prompt_template.render()}\n助手："
    # 调用模型生成答案，max_output_tokens 适当提高以避免回答被截断
    cache_key = (system_prompt, question)
    if not session_id:
        cached_answer = _cache_get(cache_key)
        if cached_answer is not None:
            meta = build_meta("cache", True, rag_meta)
            log_event(meta)
            return respond_text(cached_answer, meta=meta)

    config = LLMRequestConfig(
        temperature=req.temperature if req.temperature is not None else 0.7,
        max_output_tokens=req.max_output_tokens or 2048,
        top_p=req.top_p if req.top_p is not None else 1.0,
        top_k=req.top_k if req.top_k is not None else 1,
    )
    fallback_providers = req.fallback_providers or []

    chain = Chain(
        steps=[
            lambda state: {**state, "prompt": prompt},
        ]
    )
    chain_state = chain.run({"question": question})
    final_prompt = chain_state["prompt"]

    if not stream:
        answer, meta = llm_manager.generate(
            final_prompt,
            config=config,
            question=question,
            provider=req.provider,
            fallback=fallback_providers,
            retries=req.retries,
            enable_router=req.enable_router,
        )
        request_stats["llm"] += 1
        if not session_id:
            _cache_set(cache_key, answer)
        update_history(session_id, answer)
        log_event(build_meta("llm", False, {**meta, **rag_meta}))
        return respond_text(
            answer,
            meta=build_meta("llm", False, {**meta, **rag_meta}),
        )

    def answer_stream():
        answer_parts = []
        stream_iter, meta = llm_manager.generate_stream(
            final_prompt,
            config=config,
            question=question,
            provider=req.provider,
            fallback=fallback_providers,
            retries=req.retries,
            enable_router=req.enable_router,
        )
        for chunk in stream_iter:
            answer_parts.append(chunk)
            yield chunk
        final_answer = "".join(answer_parts)
        request_stats["llm"] += 1
        if not session_id:
            _cache_set(cache_key, final_answer)
        update_history(session_id, final_answer)
        log_event(build_meta("llm", False, {**meta, **rag_meta}))

    response = StreamingResponse(answer_stream(), media_type="text/plain; charset=utf-8")
    response.headers["X-Provider"] = req.provider or "auto"
    response.headers["X-Request-ID"] = request_id
    return response
