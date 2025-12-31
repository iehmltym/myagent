from dataclasses import dataclass          # dataclass：快速定义“配置类”，自动生成 __init__ 等
import os
from typing import Optional, Dict, Any, List  # 类型提示：让代码更清晰（运行不依赖它们）
import google.generativeai as genai        # Google Gemini SDK（官方库）
from env_utils import load_google_api_key  # 导入我们写的函数：读取 API Key


@dataclass
class GeminiConfig:
    """
    Gemini 生成参数的配置类：
    dataclass 会自动生成 __init__(temperature=..., max_output_tokens=..., ...)
    """
    temperature: float = 0.7        # 温度：越大越发散/有创意，越小越稳定/保守
    max_output_tokens: int = 1024   # 最大输出 token 数（输出越长越大）
    top_p: float = 1.0              # nucleus sampling（控制输出分布）
    top_k: int = 1                  # top-k sampling（控制候选词范围）
    # 指定模型名（如 models/gemini-1.5-flash）
    # 如果传了这个，就不会去请求模型列表（节省启动时内存和时间）
    model_name: Optional[str] = None
    # 是否自动拉取模型列表进行选择（需要额外请求，默认关闭以省内存）
    auto_select_model: bool = False


class MyGeminiLLM:
    """
    对 Gemini SDK 做一个“业务友好”的封装：
    - 自动读取 API Key
    - 支持指定模型，避免启动时拉取模型列表
    - 可选自动选择可用模型（避免 gemini-pro 404）
    - 提供统一的 generate(prompt) 方法
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        # 如果外部传了 config，就用外部的；否则使用默认 GeminiConfig()
        self.config = config or GeminiConfig()

        # 配置 SDK：把 API Key 告诉 genai（内部会用于请求鉴权）
        # load_google_api_key() 会去项目根目录的 .env 读取 GOOGLE_API_KEY
        genai.configure(api_key=load_google_api_key())

        # 指定模型优先级：config > 环境变量 > 默认值/自动选择
        # 这样可以避免启动时就拉取模型列表，降低内存占用
        self.model_name = self._resolve_model_name()

        # 注意：这里先不创建真正的模型对象（延迟到第一次生成时再创建）
        # 这样可以减少应用启动阶段的内存使用
        self.model = None

    def _resolve_model_name(self) -> str:
        """
        解析当前要使用的模型名。
        优先使用显式指定的模型，避免启动阶段大量请求/内存开销。
        """
        # 1) 优先使用代码里传入的模型名（最明确）
        if self.config.model_name:
            return self.config.model_name

        # 2) 其次使用环境变量（Render/生产环境常用方式）
        env_model = os.environ.get("GEMINI_MODEL")
        if env_model:
            return env_model

        # 3) 如果允许自动选择，再去请求模型列表（更耗资源）
        if self.config.auto_select_model:
            return self._select_available_model()

        # 4) 默认值：不请求模型列表，直接用常见的轻量模型
        return "models/gemini-1.5-flash"

    def _select_available_model(self) -> str:
        """
        从账号可用模型中选择一个支持 generateContent 的模型。
        返回值是模型名字符串（例如：models/gemini-1.5-flash）
        """

        candidates: List[str] = []  # 存放候选模型名

        # genai.list_models()：向 Google 请求“当前 API Key 可用的模型列表”
        for m in genai.list_models():
            # 每个模型对象通常有 supported_generation_methods 属性
            # 可能是 ['generateContent', ...]
            methods = getattr(m, "supported_generation_methods", []) or []

            # 只保留支持 generateContent 的模型（因为我们要用 generate_content）
            if "generateContent" in methods:
                candidates.append(m.name)

        # 如果一个都没有，说明账号权限/计费/地区限制等有问题
        if not candidates:
            raise RuntimeError("当前账号没有任何支持 generateContent 的模型。")

        # 优先选择更常用/性价比高的模型：
        # 1) gemini-1.5-flash（快、便宜）
        # 2) gemini-1.5-pro（更强）
        for prefer in ("gemini-1.5-flash", "gemini-1.5-pro"):
            for name in candidates:
                if prefer in name:
                    return name

        # 如果都不匹配，就返回列表第一个（兜底）
        return candidates[0]

    def _ensure_model(self) -> None:
        """
        确保模型对象已创建。
        只有在第一次真正生成内容时才创建，节省启动内存。
        """
        if self.model is None:
            self.model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        生成文本的统一入口。
        :param prompt: 用户输入文本
        :param kwargs: 允许调用者覆盖生成参数，例如 temperature=0.2
        :return: Gemini 返回的文本
        """

        # generation_config：本次生成的参数
        # kwargs 优先级更高：外部传了就覆盖默认 config
        generation_config: Dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_output_tokens", self.config.max_output_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
        }

        # 调用 Gemini 的生成接口：
        # - prompt：你要问的问题
        # - generation_config：控制输出风格/长度
        self._ensure_model()
        resp = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # resp.text：SDK 帮你拼好的纯文本输出（若为空则返回空字符串）
        return resp.text or ""

    def generate_stream(self, prompt: str, **kwargs: Any):
        """
        流式生成文本的统一入口。
        :param prompt: 用户输入文本
        :param kwargs: 允许调用者覆盖生成参数，例如 temperature=0.2
        :return: 逐块产出的文本片段
        """

        generation_config: Dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_output_tokens", self.config.max_output_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
        }

        self._ensure_model()
        stream = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True,
        )

        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
