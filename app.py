from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from my_llm import MyGeminiLLM

app = FastAPI()

# 允许你的 GitHub Pages 调用（先精确放行你的域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://iehmltym.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = MyGeminiLLM()

class QuestionRequest(BaseModel):
    question: str


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
    answer = llm.generate(req.question, max_output_tokens=2048)  # 避免只返回半句
    return JSONResponse({"answer": answer})
