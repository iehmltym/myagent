import os


def load_google_api_key() -> str:
    """
    直接从系统环境变量中读取 GOOGLE_API_KEY
    （Render / 本地 / Docker 通用）
    """

    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GOOGLE_API_KEY 未设置。请在 Render → Service → Environment Variables 中配置"
        )

    return api_key.strip()


# import os
# from dotenv import load_dotenv


# def load_google_api_key() -> str:
#     """
#     优先从系统环境变量读取 GOOGLE_API_KEY
#     本地开发时可通过 .env 注入
#     """

#     # 1️⃣ 先尝试加载 .env（如果存在就加载，不存在也不会报错）
#     load_dotenv()

#     # 2️⃣ 直接从环境变量中读取
#     api_key = os.getenv("GOOGLE_API_KEY")

#     # 3️⃣ 校验
#     if not api_key or not api_key.strip():
#         raise RuntimeError("未读取到 GOOGLE_API_KEY，请检查环境变量或 Render Secret")

#     return api_key.strip()



# # import os                       # os：用于获取文件路径、读取环境变量等（操作系统相关功能）
# # from dotenv import load_dotenv   # load_dotenv：从 .env 文件读取变量并注入到环境变量中


# # def load_google_api_key() -> str:
# #     """
# #     从项目根目录加载 .env 中的 GOOGLE_API_KEY，并返回该 Key。
# #     返回 str 表示这个函数一定会返回字符串。
# #     """

# #     # os.path.abspath(__file__)：得到当前文件 env_utils.py 的绝对路径（带文件名）
# #     # 例：C:\Users\HP\PycharmProjects\PythonProject23\env_utils.py
# #     current_file_abs_path = os.path.abspath(__file__)

# #     # os.path.dirname(...)：取目录部分，得到 env_utils.py 所在文件夹（也就是项目根目录）
# #     # 例：C:\Users\HP\PycharmProjects\PythonProject23
# #     base_dir = os.path.dirname(current_file_abs_path)

# #     # 拼出 .env 的绝对路径（确保无论工作目录在哪，都能找到项目根目录下的 .env）
# #     # 例：C:\Users\HP\PycharmProjects\PythonProject23\.env
# #     env_path = os.path.join(base_dir, ".env")

# #     # 读取 env_path 指向的 .env 文件，把里面的键值对加载到进程环境变量中
# #     # load_dotenv 不会“返回变量”，它只是把变量写进 os.environ
# #     load_dotenv(env_path)

# #     # os.getenv("GOOGLE_API_KEY")：从环境变量里取 GOOGLE_API_KEY
# #     # 如果没找到会返回 None
# #     api_key = os.getenv("GOOGLE_API_KEY")

# #     # 校验：如果 api_key 不存在，或者全部是空格，就抛异常
# #     # 这样能尽早告诉你“没配置 .env”，避免后续调用 API 时才报更难懂的错
# #     if not api_key or not api_key.strip():
# #         raise RuntimeError(f"未读取到 GOOGLE_API_KEY，请检查 {env_path}")

# #     # strip()：去掉前后空白（防止复制粘贴带空格导致认证失败）
# #     return api_key.strip()
