from collections import OrderedDict
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from knowledge_base import blueprint_payload, retrieve_documents
from my_custom_llm import MyCustomGeminiLLM
from tools import add, get_date, get_time, make_uuid, safe_eval_expression, text_stats

app = FastAPI()
# 会话历史字典：key 是 session_id，value 是 [(question, answer), ...]
# 用于在多轮对话中保留上下文，便于拼接成连续对话的 prompt
sessions: "OrderedDict[str, List[Tuple[str, str]]]" = OrderedDict()
MAX_HISTORY_TURNS = 10
MAX_SESSIONS = 200

answer_cache: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
MAX_CACHE_ITEMS = 256
cache_stats = {"hits": 0, "misses": 0}
request_stats = {"total": 0, "llm": 0, "rule": 0, "cache": 0}

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
DEFAULT_CUTE_SYSTEM_PROMPT = "请用可爱的语气回答，简洁、温柔，像小可爱一样～"
ADVISOR_SYSTEM_PROMPT = (
    "你是资深 AI 架构顾问，擅长 LangChain/Agent/RAG 项目落地。"
    "请用中文回答，结构必须包含："
    "1) 核心能力（LangChain 能做什么）"
    "2) 很强的典型用法（落到真实项目）"
    "3) Render 512MB 免费实例的落地建议"
    "4) 项目下一步高价值升级"
    "给出清晰的小标题和要点列表，语气专业、直接。"
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


class SessionRequest(BaseModel):
    session_id: str


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
        "limits": {
            "max_history_turns": MAX_HISTORY_TURNS,
            "max_sessions": MAX_SESSIONS,
            "cache_size": MAX_CACHE_ITEMS,
        },
        "modes": [
            {"id": "advisor", "desc": "架构建议模式：按固定结构输出升级建议"},
            {"id": "rag", "desc": "检索增强模式：结合内置资料输出"},
        ],
    }


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return {
        "requests": request_stats,
        "cache": {"size": len(answer_cache), **cache_stats},
        "sessions": {"active": len(sessions), "max": MAX_SESSIONS},
    }


@app.get("/blueprint")
def blueprint() -> Dict[str, Any]:
    return blueprint_payload()


@app.post("/session/clear")
def clear_session(req: SessionRequest) -> Dict[str, Any]:
    if sessions.pop(req.session_id, None) is not None:
        return {"cleared": True, "session_id": req.session_id}
    return {"cleared": False, "session_id": req.session_id}


def _get_session_history(session_key: Optional[str]) -> List[Tuple[str, str]]:
    if not session_key:
        return []
    history = sessions.get(session_key)
    if history is None:
        history = []
        sessions[session_key] = history
    sessions.move_to_end(session_key)
    if len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)
    return history


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


@app.post("/ask")
def ask_question(req: QuestionRequest):
    # 取出用户输入的问题文本，便于后续处理
    question = req.question.strip()
    stream = req.stream
    include_meta = req.include_meta
    forced_stream_off = False
    if include_meta and stream:
        stream = False
        forced_stream_off = True

    started = time.monotonic()
    request_stats["total"] += 1

    def respond_text(answer: str, *, meta: Optional[Dict[str, Any]] = None):
        if stream:
            def answer_stream():
                yield answer

            headers = {}
            if meta:
                headers["X-Answer-Source"] = meta.get("source", "")
                headers["X-Cache-Hit"] = str(meta.get("cache_hit", False)).lower()
            return StreamingResponse(answer_stream(), media_type="text/plain; charset=utf-8", headers=headers)
        payload = {"answer": answer}
        if include_meta and meta:
            payload["meta"] = meta
        return JSONResponse(payload)

    def update_history(session_key: Optional[str], answer: str):
        if not session_key:
            return
        history = _get_session_history(session_key)
        history.append((question, answer))
        if len(history) > MAX_HISTORY_TURNS:
            sessions[session_key] = history[-MAX_HISTORY_TURNS:]

    def build_meta(source: str, cache_hit: bool, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = {
            "source": source,
            "cache_hit": cache_hit,
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
        }
        if forced_stream_off:
            meta["stream_forced_off"] = True
        if extra:
            meta.update(extra)
        return meta

    command_result = _handle_command(question)
    if command_result:
        answer, extra = command_result
        request_stats["rule"] += 1
        return respond_text(answer, meta=build_meta("command", False, extra))
    # 规则 1：如果问题中出现“现在几点”或“今天日期”，直接返回当前时间
    if "现在几点" in question or "今天日期" in question:
        request_stats["rule"] += 1
        return respond_text(get_time(), meta=build_meta("rule", False, {"tool": "time"}))

    # 规则 2：如果问题符合 “a+b=?” 形式，解析出 a、b 并计算
    # 说明：下面这个正则允许空格和小数，比如 " 1 + 2 = ? "
    math_match = MATH_PATTERN.fullmatch(question)
    if math_match:
        # 正则分组 1 和 2 分别是 a、b 的文本形式
        left = float(math_match.group(1))
        right = float(math_match.group(2))
        # 计算完成后直接返回，避免走 LLM
        request_stats["rule"] += 1
        return respond_text(str(add(left, right)), meta=build_meta("rule", False, {"tool": "add"}))

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
    # 始终用自定义 LLM 包装器注入 system prompt，让回答保持可爱风格
    active_llm = MyCustomGeminiLLM(prefix=system_prompt)
    # 规范化 session_id（去掉首尾空白），避免同一会话被当成多个 key
    session_id = req.session_id.strip() if req.session_id else None
    # 获取该会话的历史；若不存在则初始化为空列表
    # 未提供 session_id 时视为单轮对话，不读取/写入历史
    history = _get_session_history(session_id) if session_id else []
    # 将历史记录拼成连续对话的 prompt，格式如：
    # 用户：... \n 助手：... \n 用户：... \n 助手：...
    history_prompt = "".join(
        f"用户：{user_question}\n助手：{response}\n" for user_question, response in history
    )
    # 拼接本次问题，提示模型继续回复助手内容
    if rag_context:
        prompt = (
            f"{history_prompt}用户：{req.question}\n\n"
            f"请参考以下资料作答（不要逐字复述，保持结构化输出）：\n{rag_context}\n\n"
            "助手："
        )
    else:
        prompt = f"{history_prompt}用户：{req.question}\n助手："
    # 调用模型生成答案，max_output_tokens 适当提高以避免回答被截断
    cache_key = (system_prompt, question)
    if not session_id:
        cached_answer = _cache_get(cache_key)
        if cached_answer is not None:
            return respond_text(cached_answer, meta=build_meta("cache", True, rag_meta))

    if not stream:
        answer = active_llm.generate(prompt, max_output_tokens=2048)
        request_stats["llm"] += 1
        if not session_id:
            _cache_set(cache_key, answer)
        update_history(session_id, answer)
        return respond_text(
            answer,
            meta=build_meta("llm", False, {"model": active_llm.model_name, **rag_meta}),
        )

    def answer_stream():
        answer_parts = []
        for chunk in active_llm.generate_stream(prompt, max_output_tokens=2048):
            answer_parts.append(chunk)
            yield chunk
        final_answer = "".join(answer_parts)
        request_stats["llm"] += 1
        if not session_id:
            _cache_set(cache_key, final_answer)
        update_history(session_id, final_answer)

    return StreamingResponse(answer_stream(), media_type="text/plain; charset=utf-8")
