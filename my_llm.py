from dataclasses import dataclass          # dataclass：快速定义“配置类”，自动生成 __init__ 等
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


class MyGeminiLLM:
    """
    对 Gemini SDK 做一个“业务友好”的封装：
    - 自动读取 API Key
    - 自动选择可用模型（避免 gemini-pro 404）
    - 提供统一的 generate(prompt) 方法
    """

    _configured: bool = False
    _shared_model_name: Optional[str] = None
    _shared_model: Optional[genai.GenerativeModel] = None

    def __init__(self, config: Optional[GeminiConfig] = None):
        # 如果外部传了 config，就用外部的；否则使用默认 GeminiConfig()
        self.config = config or GeminiConfig()

        if not MyGeminiLLM._configured:
            # 配置 SDK：把 API Key 告诉 genai（内部会用于请求鉴权）
            # load_google_api_key() 会去项目根目录的 .env 读取 GOOGLE_API_KEY
            genai.configure(api_key=load_google_api_key())
            MyGeminiLLM._configured = True

        if not MyGeminiLLM._shared_model_name:
            # 自动选择一个当前账号可用且支持 generateContent 的模型名
            MyGeminiLLM._shared_model_name = self._select_available_model()

        if not MyGeminiLLM._shared_model:
            # 用选择到的模型名创建模型对象
            MyGeminiLLM._shared_model = genai.GenerativeModel(MyGeminiLLM._shared_model_name)

        self.model_name = MyGeminiLLM._shared_model_name
        self.model = MyGeminiLLM._shared_model

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

        stream = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True,
        )

        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
