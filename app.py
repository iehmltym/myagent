from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

from my_llm import MyGeminiLLM

app = FastAPI()

# 允许 GitHub Pages 的域名访问（也可以先用 "*" 方便测试，上线再收紧）
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://iehmltym.github.io/"],  # 测试阶段先这样；上线建议写成你的 github.io 域名
    allow_origins=["*"],  # 测试阶段先这样；上线建议写成你的 github.io 域名
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = MyGeminiLLM()

class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(req: QuestionRequest):
    answer = llm.generate(req.question, max_output_tokens=2048)
    return JSONResponse({"answer": answer})


# from fastapi import FastAPI
# from fastapi.responses import HTMLResponse, JSONResponse
# from pydantic import BaseModel
#
# # 引入你已经写好的 LLM
# from my_llm import MyGeminiLLM
#
# # 1️⃣ 创建 FastAPI 应用
# app = FastAPI()
#
# # 2️⃣ 创建一个 LLM 实例（整个程序共用一个）
# llm = MyGeminiLLM()
#
# # 3️⃣ 定义“前端传过来的数据格式”
# class QuestionRequest(BaseModel):
#     question: str
#
#
# # 4️⃣ 首页接口：返回 HTML 页面
# @app.get("/", response_class=HTMLResponse)
# def index():
#     # 直接返回一个 HTML 文件内容
#     with open("templates/index.html", "r", encoding="utf-8") as f:
#         return f.read()
#
#
# # 5️⃣ 给前端用的 API 接口
# @app.post("/ask")
# def ask_question(req: QuestionRequest):
#     """
#     前端会 POST 一个 JSON：
#     {
#         "question": "你好"
#     }
#     """
#     answer = llm.generate(req.question)
#
#     # 返回 JSON 给前端
#     return JSONResponse({
#         "answer": answer
#     })
