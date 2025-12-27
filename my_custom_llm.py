from typing import Optional, Any             # 类型提示
from my_llm import MyGeminiLLM, GeminiConfig # 复用基础封装和配置类


class MyCustomGeminiLLM(MyGeminiLLM):
    """
    自定义 LLM：在基础 LLM 上增加“固定前缀”
    这个前缀相当于：系统提示词 / 角色设定 / 固定上下文
    """

    def __init__(self, prefix: str, config: Optional[GeminiConfig] = None):
        # 把 prefix 存起来，strip 去掉首尾空白
        self.prefix = prefix.strip()

        # 调用父类初始化：加载 key、选模型、创建 model
        super().__init__(config=config)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        # 如果 prefix 非空，把 prefix + 用户输入拼成最终 prompt
        if self.prefix:
            final_prompt = f"{self.prefix}\n\n用户：{prompt}\n助手："
        else:
            final_prompt = prompt

        # 调用父类 generate 实际请求 Gemini
        return super().generate(final_prompt, **kwargs)
