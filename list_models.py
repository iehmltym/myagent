from google import genai
from env_utils import load_google_api_key   # 读取 API Key


def main():
    # 配置 SDK 鉴权
    client = genai.Client(api_key=load_google_api_key())

    # list_models()：打印当前账号可用模型
    for m in client.models.list():
        # m.name：模型名（例如 models/gemini-1.5-flash）
        # supported_generation_methods：支持的方法列表
        print(m.name, getattr(m, "supported_generation_methods", None))


if __name__ == "__main__":
    # 只有当你直接运行 python list_models.py 时，才会执行 main()
    main()
