import logging

# 配置 logging
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO，也可以改为 DEBUG
    format=
    '%(asctime)s - %(module)s - %(funcName)s - %(lineno)s - %(message)s',  # 设置日志格式
    datefmt='%m-%d %H:%M:%S'  # 设置时间格式
)

# 设置所有与 HTTP 请求相关的日志级别为 WARNING
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("SentenceTransformer").setLevel(logging.WARNING)
# 创建一个 logger 对象
logger = logging.getLogger(__name__)
