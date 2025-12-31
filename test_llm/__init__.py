from my_llm import MyGeminiLLM, GeminiConfig          # 基础 LLM 和配置
from my_custom_llm import MyCustomGeminiLLM           # 带前缀的 LLM


def main():
    # ===== 1) 测试基础 LLM =====
    llm = MyGeminiLLM(
        GeminiConfig(
            temperature=0.7,        # 输出相对自然
            max_output_tokens=512   # 输出不宜太长，便于观察
        )
    )

    # 调用 generate 生成文本
    print(llm.generate("用中文解释什么是 中华美食，并给一个简单例子。"))

    # ===== 2) 测试带前缀的 LLM =====
    prefix = "你是一名资深后端工程师，回答要结构化，包含概念、要点、示例。"

    # 创建带前缀的 LLM
    llm2 = MyCustomGeminiLLM(prefix)

    # 生成文本：会自动把 prefix 拼到 prompt 前面
    print(llm2.generate("解释 DTO 和 Map 的区别，并说明常见互转方式。"))


if __name__ == "__main__":
    # python -m test_llm 或直接运行本文件时，会从这里开始
    main()
