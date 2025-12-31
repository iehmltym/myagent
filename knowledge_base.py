from __future__ import annotations

from typing import Any, Dict, List

KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "core-llm",
        "title": "LLM/ChatModel 抽象层（Provider 无关）",
        "category": "core",
        "content": (
            "同一套接口支持 OpenAI / Gemini / Anthropic / 本地模型（Ollama），"
            "可统一管理温度、token、system prompt、工具调用、重试和回退策略。"
            "价值：把模型供应商变化对业务代码的影响降到最低。"
        ),
        "keywords": ["llm", "chatmodel", "provider", "openai", "gemini", "anthropic", "ollama", "抽象层"],
    },
    {
        "id": "core-prompt",
        "title": "Prompt 体系（PromptTemplate + Messages）",
        "category": "core",
        "content": (
            "结构化 prompt 支持变量填充、消息角色、few-shot，并且可以组件化复用、版本化管理。"
            "价值：比手写字符串稳定，可测试可迭代。"
        ),
        "keywords": ["prompt", "messages", "few-shot", "模板", "角色", "提示词"],
    },
    {
        "id": "core-chains",
        "title": "Chains（流程编排）",
        "category": "core",
        "content": (
            "支持用户输入→预处理→检索→生成→后处理的流水线，"
            "并能做分支/并行/条件路由。"
            "价值：把业务流程从散落的 if/else 中抽离，逻辑更清晰。"
        ),
        "keywords": ["chains", "workflow", "编排", "流程", "分支", "路由"],
    },
    {
        "id": "core-rag",
        "title": "RAG 组件（检索增强生成）",
        "category": "core",
        "content": (
            "Document Loaders + Text Splitters + Embeddings + VectorStore/Retriever 组合，"
            "解决模型不知道私有知识的问题，是生产落地最常见模式。"
        ),
        "keywords": ["rag", "retriever", "vector", "embedding", "faiss", "chroma", "检索"],
    },
    {
        "id": "core-agents",
        "title": "Tools & Agents（工具调用 + 自动决策）",
        "category": "core",
        "content": (
            "Tools 用来封装函数/HTTP API/数据库查询；Agents 让模型自动决定调用哪个工具。"
            "价值：从聊天升级为能执行任务的控制器。"
        ),
        "keywords": ["agent", "tools", "function", "tool calling", "调用"],
    },
    {
        "id": "core-multimodal",
        "title": "多模态输入（文本/图片/文件/语音）",
        "category": "core",
        "content": (
            "统一接收图片、文件、音频输入，抽取结构化摘要再喂给模型。"
            "价值：让 Agent 处理真实业务素材（合同、截图、语音记录）。"
        ),
        "keywords": ["multimodal", "多模态", "图片", "文件", "语音", "audio"],
    },
    {
        "id": "core-toolkit",
        "title": "工具箱能力（文本清洗/抽取/JSON/关键词）",
        "category": "core",
        "content": (
            "提供文本清洗、URL/邮箱抽取、Markdown 大纲、关键词统计等基础工具。"
            "价值：让 LLM 更像业务处理器，输出可直接下游使用的数据。"
        ),
        "keywords": ["toolkit", "extract", "keyword", "markdown", "json"],
    },
    {
        "id": "core-memory",
        "title": "Memory / State（对话记忆）",
        "category": "core",
        "content": (
            "短期记忆保留最近 N 轮，长期记忆把历史对话向量化存储按需检索。"
            "价值：更像有上下文的助手，但要控制成本与隐私。"
        ),
        "keywords": ["memory", "state", "对话", "历史", "window", "long-term"],
    },
    {
        "id": "use-agent",
        "title": "工具调用 Agent = 最值钱的“控制器”能力",
        "category": "use",
        "content": (
            "自动判断查数据库还是直接回答，能分解任务并执行多步流程。"
            "适合客服、运维助手、代码分析、自动化工作台，但多轮调用会消耗 token 和延迟。"
        ),
        "keywords": ["agent", "tools", "控制器", "自动化", "运维", "客服"],
    },
    {
        "id": "use-multimodal",
        "title": "多模态场景落地",
        "category": "use",
        "content": (
            "文件上传 → 解析元数据 → 结构化摘要 → 进入 RAG/Agent 流程。"
            "适合合同审阅、图片质检、语音会议纪要。"
        ),
        "keywords": ["多模态", "文件上传", "图片", "语音", "摘要"],
    },
    {
        "id": "use-rag",
        "title": "RAG 工程化 = 更稳定的“有知识的回答”",
        "category": "use",
        "content": (
            "优化 chunk 策略、top-k、rerank、结构化上下文注入。"
            "知识库可更新，不必重新训练模型。"
        ),
        "keywords": ["rag", "chunk", "top-k", "rerank", "知识库"],
    },
    {
        "id": "use-router",
        "title": "Router 与多模型回退 = 成本控制利器",
        "category": "use",
        "content": (
            "简单问题走便宜模型，复杂问题走强模型；检索失败走澄清链。"
            "这是成本控制与稳定性最有效的工程手段之一。"
        ),
        "keywords": ["router", "fallback", "成本", "回退", "模型选择"],
    },
    {
        "id": "use-stream",
        "title": "Streaming 输出 = 体验与延迟双赢",
        "category": "use",
        "content": (
            "首 token 更快，减少等待感，尤其适合 Render 免费实例。"
            "注意处理超时与反向代理的流式兼容性。"
        ),
        "keywords": ["stream", "streaming", "流式", "延迟"],
    },
    {
        "id": "use-observe",
        "title": "Observability = 生产调试必备",
        "category": "use",
        "content": (
            "通过 trace、token、latency、工具调用序列定位回答质量问题。"
            "免费实例可先用轻量日志代替全量平台。"
        ),
        "keywords": ["observability", "trace", "latency", "日志", "langsmith"],
    },
    {
        "id": "render-limits",
        "title": "Render 512MB 免费实例的现实瓶颈",
        "category": "render",
        "content": (
            "内存瓶颈主要在依赖体积、向量库常驻内存、文档处理峰值和并发。"
            "策略：尽量轻量依赖、外部托管向量库、离线处理文档、控制并发。"
        ),
        "keywords": ["render", "512mb", "内存", "依赖", "向量库", "并发"],
    },
    {
        "id": "render-route",
        "title": "最适合 512MB 的轻量落地路线",
        "category": "render",
        "content": (
            "先做 RAG + 流式输出，Agent 放后面；"
            "小规模数据优先外部向量库（Pinecone/Weaviate/PGVector/Upstash）。"
        ),
        "keywords": ["512mb", "rag", "stream", "向量库", "外部托管"],
    },
    {
        "id": "render-memory",
        "title": "Memory 优先用窗口记忆",
        "category": "render",
        "content": (
            "先保留最近 6~10 轮对话，长期记忆向量化后续再上。"
            "先控成本再扩展。"
        ),
        "keywords": ["memory", "窗口", "成本", "512mb"],
    },
    {
        "id": "next-steps",
        "title": "下一步升级优先级",
        "category": "next",
        "content": (
            "RAG（固定数据源→离线 embedding→在线检索→stream 输出）"
            "→ Router（简单问答不检索）"
            "→ 轻量 observability（请求日志 + top-k 命中）"
            "→ 小范围 Agent 工具。"
        ),
        "keywords": ["升级", "roadmap", "优先级", "下一步"],
    },
]


def retrieve_documents(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    query_lower = query.lower().strip()
    if not query_lower:
        return []
    scored: List[Dict[str, Any]] = []
    for doc in KNOWLEDGE_BASE:
        score = 0
        for keyword in doc["keywords"]:
            if keyword.lower() in query_lower:
                score += 2
        if doc["title"] in query:
            score += 1
        if score:
            scored.append({"score": score, **doc})
    if not scored:
        return [doc for doc in KNOWLEDGE_BASE if doc["category"] in {"core", "use", "render"}][:top_k]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return [item for item in scored[:top_k]]


def blueprint_payload() -> Dict[str, Any]:
    sections: Dict[str, List[Dict[str, str]]] = {
        "core": [],
        "use": [],
        "render": [],
        "next": [],
    }
    for doc in KNOWLEDGE_BASE:
        if doc["category"] in sections:
            sections[doc["category"]].append(
                {
                    "title": doc["title"],
                    "content": doc["content"],
                }
            )
    return {
        "sections": sections,
        "tips": [
            "先把知识库离线 embedding，线上只做检索 + 生成",
            "优先外部托管向量库，把内存压力移出 Render",
            "短期记忆先用窗口记忆（6~10 轮）",
        ],
    }
