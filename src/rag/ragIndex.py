import os
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope

# =================配置区域=================
# 假设你的项目结构如下，请根据实际路径调整
BASE_DIR = Path(__file__).parent
DOC_PATH = BASE_DIR / "data/光端上云客服诊断话术梳理.xlsx"
# 定义数据库目录路径
DB_DIR = BASE_DIR / "embeddingDB"
# 定义具体的 db 文件路径 (Milvus Lite 需要这个文件路径)
MILVUS_URI = str(DB_DIR / "milvus_service.db")
# 定义索引持久化目录
PERSIST_DIR = str(DB_DIR / "doc_emb")

COLLECTION_NAME = "fault_table"
VEC_DIM = 1024


# =================1. 解决错误1：确保目录存在=================
def ensure_directories():
    """
    在初始化 Milvus 之前，必须确保目录存在。
    否则 Milvus Lite 会抛出 ConnectionConfigException: dir not exists
    """
    if not DB_DIR.exists():
        print(f"📁 创建缺失的目录: {DB_DIR}")
        DB_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(PERSIST_DIR).exists():
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)


# =================2. 初始化模型=================
def init_models():
    Settings.llm = DashScope(
        model="qwen-max",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL")
    )
    Settings.embed_model = DashScopeEmbedding(
        model="text-embedding-v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL")
    )


# =================3. 获取向量存储（解决错误2的关键）=================
def get_vector_store():
    """
    初始化 MilvusVectorStore。
    关键点：
    1. uri 指向已确保存在的 db 文件路径。
    2. auto_load=True 确保集合在搜索前自动加载到内存，解决 'released' 错误。
    """
    return MilvusVectorStore(
        uri=MILVUS_URI,
        collection_name=COLLECTION_NAME,
        dim=VEC_DIM,
        auto_load=True,  # 【核心】自动处理 load()，无需手动干预
        overwrite=False
    )


# =================4. 构建或加载索引=================
def build_or_load_index():
    # 第一步：确保目录存在（解决错误1）
    ensure_directories()

    # 第二步：获取向量存储（配置了 auto_load）
    vector_store = get_vector_store()

    # 第三步：尝试从磁盘加载索引
    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR,
                vector_store=vector_store
            )
            index = VectorStoreIndex.from_storage(storage_context)
            print("✅ 成功加载现有索引")
            return index
        except Exception as e:
            print(f"⚠️ 加载索引失败: {e}，将重建索引")

    # 第四步：如果不存在或加载失败，则新建索引
    # 这里替换为你实际的数据加载逻辑
    from src.rag.docChunk import load_fault_excel  # 假设你有这个函数
    documents = load_fault_excel(DOC_PATH)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        vector_store=vector_store,
        show_progress=True
    )

    # 持久化保存
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("✅ 新索引构建并保存成功")

    return index


# =================5. 执行查询=================
def query_knowledge( fault_code: str = None):
    """
    执行查询。
    由于 auto_load=True 且目录已预创建，这里不会报错。
    """
    init_models()
    index = build_or_load_index()
    query_engine = index.as_retriever(similarity_top_k=2)

    nodes = query_engine.retrieve(fault_code)
    if not nodes:
        return "未查询到对应故障处理方案，请转接人工客服。"

    fault_text = "\n=====【官方故障处理规范】=====\n"
    for node in nodes:
        fault_text += f"\n【匹配相似度：{node.score:.2f}】\n{node.text}\n"
    return fault_text


# if __name__ == "__main__":
#     # 初始化环境变量等
#     init_models()
#
#     # 测试查询
#     try:
#         result = query_knowledge("100")
#         print("\n回答:", result)
#     except Exception as e:
#         print(f"\n❌ 发生错误: {e}")
