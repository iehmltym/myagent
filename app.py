from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from my_llm import MyGeminiLLM
from my_custom_llm import MyCustomGeminiLLM
from tools import add, get_time

app = FastAPI()
# 会话历史字典：key 是 session_id，value 是 [(question, answer), ...]
# 用于在多轮对话中保留上下文，便于拼接成连续对话的 prompt
sessions: Dict[str, List[Tuple[str, str]]] = {}

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

# 全局默认的 LLM 实例（不带 system prompt 的情况使用它）
llm = MyGeminiLLM()
# 默认的可爱语气 system prompt（当用户未提供时使用）
DEFAULT_CUTE_SYSTEM_PROMPT = "请用可爱的语气回答，简洁、温柔，像小可爱一样～"

class QuestionRequest(BaseModel):
    # 用户输入的问题文本
    question: str
    # 可选的 system prompt，用来影响模型输出风格
    # 如果不传或传空字符串，就会走默认的 MyGeminiLLM
    system_prompt: Optional[str] = None
    # 会话 ID：用于区分不同用户/对话的上下文
    # 为空时表示单轮请求，不记录历史
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
    </body>
    </html>
    """


@app.post("/ask")
def ask_question(req: QuestionRequest):
    # 取出用户输入的问题文本，便于后续处理
    question = req.question
    # 规则 1：如果问题中出现“现在几点”或“今天日期”，直接返回当前时间
    if "现在几点" in question or "今天日期" in question:
        return JSONResponse({"answer": get_time()})

    # 规则 2：如果问题符合 “a+b=?” 形式，解析出 a、b 并计算
    # 说明：下面这个正则允许空格和小数，比如 " 1 + 2 = ? "
    math_match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*=\s*\?\s*", question)
    if math_match:
        # 正则分组 1 和 2 分别是 a、b 的文本形式
        left = float(math_match.group(1))
        right = float(math_match.group(2))
        # 计算完成后直接返回，避免走 LLM
        return JSONResponse({"answer": str(add(left, right))})

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
    answer = active_llm.generate(prompt, max_output_tokens=2048)  # 避免只返回半句
    if session_id:
        # 仅当 session_id 有效时才记录历史，避免无意义的全局堆积
        history.append((req.question, answer))
    # 返回格式保持 {"answer": ...}，确保前端兼容
    return JSONResponse({"answer": answer})

