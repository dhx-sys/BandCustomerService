
import os


class Config:
    """统一的配置类，集中管理所有常量"""
    # prompt文件路径
    PROMPT_INTENT_RECOGNIZE = "prompts/intent_recognize2.txt"
    PROMPT_IS_SAME_PACKAGE_USER = "prompts/is_same_package_user.txt"
    PROMPT_CHECK_ID_CARD = "prompts/check_id_card.txt"
    PROMPT_MATCHED_BAND_ADDRESS = "prompts/matched_band_address1.txt"
    PROMPT_FAULT_MANAGE = "prompts/fault_manage.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # PostgreSQL数据库配置参数
    DB_URI = os.getenv("DB_URI", "postgresql://postgres:dhx2589630@127.0.0.1:5432/postgres?sslmode=disable")
    MIN_SIZE = 1
    MAX_SIZE = 2

    # 日志持久化存储
    LOG_FILE = "logfile/app.log"
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    MAX_BYTES=5*1024*1024,
    BACKUP_COUNT=3


    # openai:调用gpt模型, qwen:调用阿里通义千问大模型, oneapi:调用oneapi方案支持的模型, Lmdeploy:调用本地开源大模型
    LLM_TYPE = "qwen"
    #LLM_TYPE = "openaideepseek"


    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    SESSION_TIMEOUT = 3600
    TTL = 3600
    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012