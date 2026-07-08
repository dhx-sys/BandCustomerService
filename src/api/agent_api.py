import sys
import os
from langsmith import Client, traceable

# LangSmith 全局配置
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "宽带故障客服机器人"

# 实例化客户端，用于手动创建反馈、日志查询
langsmith_client = Client()
# 关键修复步骤：必须在导入 uvicorn 或创建 app 之前设置
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from uvicorn import Config
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from psycopg_pool import AsyncConnectionPool
from src.agent.workflow import create_graph
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from pydantic import BaseModel, Field
import time
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional, List
import uuid
from contextlib import asynccontextmanager
import redis.asyncio as redis
import json
from datetime import timedelta
from config import Config
from llms import get_llm
from langgraph.types import Command

# 设置日志基本配置，级别为DEBUG或INFO
logger = logging.getLogger(__name__)
# 设置日志器级别为DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # 清空默认处理器
# 使用ConcurrentRotatingFileHandler
handler = ConcurrentRotatingFileHandler(
    # 日志文件
    Config.LOG_FILE,
    # 日志文件最大允许大小为5MB，达到上限后触发轮转
    maxBytes = Config.MAX_BYTES,
    # 在轮转时，最多保留3个历史日志文件
    backupCount = Config.BACKUP_COUNT
)
# 设置处理器级别为DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

# 定义数据模型 客户端发起的运行智能体的请求数据
class AgentRequest(BaseModel):
    # 用户唯一标识
    user_id: str
    # 会话唯一标识
    session_id: str
    # 用户的问题
    query: str

# 定义数据模型 运行智能体后返回的响应数据
class AgentResponse(BaseModel):
    # 会话唯一标识
    session_id: str
    # 三个状态：interrupted, completed, error
    status: str
    # 时间戳
    timestamp: float = Field(default_factory=lambda: time.time())
    # error时的提示消息
    message: Optional[str] = None
    # completed时的结果消息
    result: Optional[Dict[str, Any]] = None
    # interrupted时的中断消息
    interrupt_data: Optional[Dict[str, Any]] = None

# 处理智能体返回结果 可能是中断，也可能是最终结果
async def process_agent_result(
        session_id: str,
        result: Dict[str, Any],
        user_id: Optional[str] = None
) -> AgentResponse:
    """
    处理智能体执行结果，统一处理中断和结果

    Args:
        session_id: 会话ID
        result: 智能体执行结果
        user_id: 用户ID，如果提供，将更新会话状态

    Returns:
        AgentResponse: 标准化的响应对象
    """
    response = None

    try:
        # 检查是否有中断

        if result.interrupts:
            interrupt_data  = result.interrupts[0].value
            # # 确保中断数据有类型信息
            # if "interrupt_type" not in interrupt_data:
            #     interrupt_data["interrupt_type"] = "unknown"
            # 返回中断信息
            response = AgentResponse(
                session_id=session_id,
                status="interrupted",
                interrupt_data=interrupt_data
            )
            # #将中断时AI回复的消息保存到state中
            await app.state.agent.aupdate_state(
                {"configurable": {"thread_id": session_id}},
                {"messages": [AIMessage(interrupt_data["content"])]}
            )
            logger.info(f"当前触发工具调用中断:{response}")
        # 如果没有中断，返回最终结果
        else:
            response = AgentResponse(
                session_id=session_id,
                status="completed",
                result=result
            )
            logger.info(f"最终智能体回复结果:{response}")

    except Exception as e:
        response = AgentResponse(
            session_id=session_id,
            status="error",
            message=f"处理智能体结果时出错: {str(e)}"
        )
        logger.error(f"处理智能体结果时出错:{response}")

    # 若会话存在，更新会话状态
    exists = await app.state.session_manager.session_id_exists(user_id, session_id)
    if exists:
        status = response.status
        last_query = None
        last_response = response
        last_updated = time.time()
        ttl = Config.TTL
        await app.state.session_manager.update_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

    return response

# 实现redis相关方法 支持多用户多会话
class RedisSessionManager:
    # 初始化 RedisSessionManager 实例
    # 配置 Redis 连接参数和默认会话超时时间
    def __init__(self, redis_host: str, redis_port: int, redis_db: int, session_timeout: int):
        # 创建 Redis 客户端连接
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        # 设置默认会话过期时间（秒）
        self.session_timeout = session_timeout

    # 关闭 Redis 连接
    async def close(self):
        # 异步关闭 Redis 客户端连接
        await self.redis_client.aclose()

    # 创建指定用户的新会话
    # 存储结构：session:{user_id}:{session_id} = {
    #   "session_id": session_id,
    #   "status": "idle|running|interrupted|completed|error",
    #   "last_response": AgentResponse,
    #   "last_query": str,
    #   "last_updated": timestamp
    # }
    async def create_session(self, user_id: str, session_id: Optional[str] = None, status: str = "active",
                            last_query: Optional[str] = None, last_response: Optional['AgentResponse'] = None,
                            last_updated: Optional[float] = None, ttl: Optional[int] = None) -> str:
        # 如果未提供 session_id，生成新的 UUID
        if session_id is None:
            session_id = str(uuid.uuid4())
        # 如果未提供最后更新时间，设置为 0 秒
        if last_updated is None:
            last_updated = str(timedelta(seconds=0))
        # 使用提供的 TTL 或默认的 session_timeout
        effective_ttl = ttl if ttl is not None else self.session_timeout

        # 构造会话数据结构
        session_data = {
            "session_id": session_id,
            "status": status,
            "last_response": last_response.model_dump() if isinstance(last_response, BaseModel) else last_response,
            "last_query": last_query,
            "last_updated": last_updated
        }

        # 将会话数据存储到 Redis，使用 JSON 序列化，并设置过期时间
        await self.redis_client.set(
            f"session:{user_id}:{session_id}",
            json.dumps(session_data, default=lambda o: o.__dict__ if not hasattr(o, 'model_dump') else o.model_dump()),
            ex=effective_ttl
        )
        # 将 session_id 添加到用户的会话列表中
        await self.redis_client.sadd(f"user_sessions:{user_id}", session_id)
        # 返回新创建的 session_id
        return session_id

    # 更新指定用户的特定会话数据
    async def update_session(self, user_id: str, session_id: str, status: Optional[str] = None,
                            last_query: Optional[str] = None, last_response: Optional['AgentResponse'] = None,
                            last_updated: Optional[float] = None, ttl: Optional[int] = None) -> bool:
        # 检查会话是否存在
        if await self.redis_client.exists(f"session:{user_id}:{session_id}"):
            # 获取当前会话数据
            current_data = await self.get_session(user_id, session_id)
            if not current_data:
                return False
            # 更新提供的字段
            if status is not None:
                current_data["status"] = status
            if last_response is not None:
                if isinstance(last_response, BaseModel):
                    current_data["last_response"] = last_response.model_dump()
                else:
                    current_data["last_response"] = last_response
            if last_query is not None:
                current_data["last_query"] = last_query
            if last_updated is not None:
                current_data["last_updated"] = last_updated
            # 使用提供的 TTL 或默认的 session_timeout
            effective_ttl = ttl if ttl is not None else self.session_timeout
            # 将更新后的数据重新存储到 Redis，并设置新的过期时间
            await self.redis_client.set(
                f"session:{user_id}:{session_id}",
                json.dumps(current_data,
                           default=lambda o: o.__dict__ if not hasattr(o, 'model_dump') else o.model_dump()),
                ex=effective_ttl
            )
            # 更新成功返回 True
            return True
        # 会话不存在返回 False
        return False

    # 获取指定用户当前会话ID的状态数据
    async def get_session(self, user_id: str, session_id: str) -> Optional[dict]:
        # 从 Redis 获取会话数据
        session_data = await self.redis_client.get(f"session:{user_id}:{session_id}")
        # 如果会话不存在，返回 None
        if not session_data:
            return None
        # 解析 JSON 数据
        session = json.loads(session_data)
        # 处理 last_response 字段，尝试转换为 AgentResponse 对象
        if session and "last_response" in session:
            if session["last_response"] is not None:
                try:
                    session["last_response"] = AgentResponse(**session["last_response"])
                except Exception as e:
                    # 记录转换失败的错误日志
                    logger.error(f"转换 last_response 失败: {e}")
                    session["last_response"] = None
        # 返回会话数据
        return session

    # 获取指定用户下的当前激活的会话ID
    async def get_user_active_session_id(self, user_id: str) -> str | None:
        # 在查询前清理指定用户的无效会话
        await self.cleanup_user_sessions(user_id)

        # 获取用户的所有 session_id
        session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")

        # 初始化最新会话信息
        latest_session_id = None
        latest_timestamp = -1  # 使用负值确保任何有效时间戳都更大

        # 遍历每个 session_id，获取会话数据
        for session_id in session_ids:
            session = await self.get_session(user_id, session_id)
            if session:
                last_updated = session.get('last_updated')
                # 过滤掉 last_updated 为 "0:00:00" 的记录
                if isinstance(last_updated, str) and last_updated == "0:00:00":
                    continue
                # 确保 last_updated 是数字（时间戳）
                if isinstance(last_updated, (int, float)) and last_updated > latest_timestamp:
                    latest_timestamp = last_updated
                    latest_session_id = session_id

        # 返回最新会话ID，如果没有有效会话则返回 None
        return latest_session_id

    # 获取指定用户下的所有 session_id
    async def get_all_session_ids(self, user_id: str) -> List[str]:
        # 在查询前清理指定用户的无效会话，确保返回的 session_id 都是有效的
        await self.cleanup_user_sessions(user_id)
        # 从 Redis 获取用户的所有 session_id
        session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
        # 将集合转换为列表并返回
        return list(session_ids)

    # 获取系统内所有用户下的所有 session_id
    async def get_all_users_session_ids(self) -> Dict[str, List[str]]:
        # 清理所有用户的无效会话
        await self.cleanup_all_sessions()
        # 初始化结果字典
        result = {}
        # 遍历所有 user_sessions:* 键
        async for key in self.redis_client.scan_iter("user_sessions:*"):
            # 提取用户 ID
            user_id = key.split(":", 1)[1]
            # 获取该用户的所有 session_id
            session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
            # 如果集合非空，将用户 ID 和 session_id 列表存入结果字典
            if session_ids:
                result[user_id] = list(session_ids)
        # 返回所有用户及其 session_id
        return result

    # 获取指定用户ID的所有会话状态详情数据
    async def get_all_user_sessions(self, user_id: str) -> List[dict]:
        # 初始化会话列表
        sessions = []
        # 获取用户的所有 session_id
        session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
        # 遍历每个 session_id，获取会话数据
        for session_id in session_ids:
            session = await self.get_session(user_id, session_id)
            if session:
                sessions.append(session)
        # 返回所有会话数据
        return sessions

    # 检查指定用户ID是否在 Redis 中
    async def user_id_exists(self, user_id: str) -> bool:
        # 在查询前清理指定用户的无效会话
        await self.cleanup_user_sessions(user_id)
        # 检查是否存在 user_sessions:{user_id} 键
        return (await self.redis_client.exists(f"user_sessions:{user_id}")) > 0

    # 检查指定用户ID的特定 session_id 是否存在
    async def session_id_exists(self, user_id: str, session_id: str) -> bool:
        # 在查询前清理指定用户的无效会话
        await self.cleanup_user_sessions(user_id)
        # 检查指定用户的特定会话是否存在
        return (await self.redis_client.exists(f"session:{user_id}:{session_id}")) > 0

    # 获取所有会话数量
    async def get_session_count(self) -> int:
        # 清理所有用户的无效会话
        await self.cleanup_all_sessions()
        # 初始化计数器
        count = 0
        # 遍历所有 session:* 键
        async for _ in self.redis_client.scan_iter("session:*"):
            count += 1
        # 返回会话总数
        return count

    # 清理指定用户的无效会话
    async def cleanup_user_sessions(self, user_id: str) -> None:
        # 获取用户会话集合中的所有 session_id
        session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
        # 遍历每个 session_id，检查对应的会话键是否存在
        for session_id in session_ids:
            if not await self.redis_client.exists(f"session:{user_id}:{session_id}"):
                # 如果会话键已过期或不存在，从集合中移除 session_id
                await self.redis_client.srem(f"user_sessions:{user_id}", session_id)
                logger.info(f"Removed expired session_id {session_id} for user {user_id}")
        # 如果集合为空，删除集合
        if not await self.redis_client.scard(f"user_sessions:{user_id}"):
            await self.redis_client.delete(f"user_sessions:{user_id}")
            logger.info(f"Deleted empty user_sessions collection for user {user_id}")

    # 清理所有用户的无效会话
    async def cleanup_all_sessions(self) -> None:
        # 遍历所有 user_sessions:* 键
        async for key in self.redis_client.scan_iter("user_sessions:*"):
            # 提取用户 ID
            user_id = key.split(":", 1)[1]
            # 获取用户会话集合中的所有 session_id
            session_ids = await self.redis_client.smembers(f"user_sessions:{user_id}")
            # 遍历每个 session_id，检查对应的会话键是否存在
            for session_id in session_ids:
                if not await self.redis_client.exists(f"session:{user_id}:{session_id}"):
                    # 如果会话键已过期或不存在，从集合中移除 session_id
                    await self.redis_client.srem(f"user_sessions:{user_id}", session_id)
                    logger.info(f"Removed expired session_id {session_id} for user {user_id}")
            # 如果集合为空，删除集合
            if not await self.redis_client.scard(f"user_sessions:{user_id}"):
                await self.redis_client.delete(f"user_sessions:{user_id}")
                logger.info(f"Deleted empty user_sessions collection for user {user_id}")

    # 删除指定用户的特定会话
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        # 从用户会话列表中移除 session_id
        await self.redis_client.srem(f"user_sessions:{user_id}", session_id)
        # 删除会话数据并返回是否成功
        return (await self.redis_client.delete(f"session:{user_id}:{session_id}")) > 0
# 读取指定用户长期记忆中的内容
async def read_long_term_info(user_id :str):
    """
    读取指定用户长期记忆中的内容

    Args:
        user_id: 用户的唯一标识

    Returns:
        Dict[str, Any]: 包含记忆内容和状态的响应
    """
    try:
        # 指定命名空间
        namespace = ("memories", user_id)

        # 搜索记忆内容
        memories = await app.state.store.asearch(namespace, query="")

        # 处理查询结果
        if memories is None:
            raise HTTPException(
                status_code=500,
                detail="查询返回无效结果，可能是存储系统错误。"
            )

        # 提取并拼接记忆内容
        long_term_info = " ".join(
            [d.value["data"] for d in memories if isinstance(d.value, dict) and "data" in d.value]
        ) if memories else ""


        # 记录查询成功的日志
        logger.info(f"成功获取用户ID: {user_id} 的长期记忆，内容长度: {len(long_term_info)} 字符")

        # 返回结构化响应
        return {
            "success": True,
            "user_id": user_id,
            "long_term_info": long_term_info,
            "message": "长期记忆获取成功" if long_term_info else "未找到长期记忆内容"
        }

    except Exception as e:
        # 处理其他未预期的错误
        logger.error(f"获取用户ID: {user_id} 的长期记忆时发生意外错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取长期记忆失败: {str(e)}"
        )
# 生命周期函数 app应用初始化函数
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 实例化异步Redis会话管理器 并存储为单实例
        app.state.session_manager = RedisSessionManager(
            Config.REDIS_HOST,
            Config.REDIS_PORT,
            Config.REDIS_DB,
            Config.SESSION_TIMEOUT
        )
        logger.info("Redis初始化成功")
        async with AsyncConnectionPool(
                conninfo=Config.DB_URI,
                min_size=Config.MIN_SIZE,
                max_size=Config.MAX_SIZE,
                kwargs={"autocommit": True, "prepare_threshold": 0}
        ) as pool:
            # 长期记忆 初始化checkpointer，并初始化表结构
            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()
            logger.info("Checkpointer初始化成功")

            # 创建Chat模型
            llm_chat = get_llm(Config.LLM_TYPE)
            logger.info("Chat模型初始化成功")

            logger.info("启动：构建LangGraph智能体流程图")
            agent_graph = create_graph(llm_chat, checkpointer=checkpointer)
            app.state.agent = agent_graph
            logger.info("Agent初始化成功")

            logger.info("服务完成初始化并启动服务")
            yield
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"服务初始化失败: {str(e)}")

    # 清理资源
    finally:
        # 关闭Redis连接
        if hasattr(app.state, 'session_manager'):
            await app.state.session_manager.close()
        logger.info("关闭服务并完成资源清理")

# 实例化app 并使用生命周期上下文管理器进行app初始化
app = FastAPI(
    title="Agent智能体后端API接口服务",
    description="基于LangGraph提供AI Agent服务",
    lifespan=lifespan
)
# 解析state消息列表进行格式化展示
async def parse_messages(messages: List[Any]) -> None:
    """
    解析消息列表，打印 HumanMessage、AIMessage 和 ToolMessage 的详细信息

    Args:
        messages: 包含消息的列表，每个消息是一个对象
    """
    print("=== 消息解析结果 ===")
    for idx, msg in enumerate(messages, 1):
        print(f"\n消息 {idx}:")
        # 获取消息类型
        msg_type = msg.__class__.__name__
        print(f"类型: {msg_type}")
        # 提取消息内容
        content = getattr(msg, 'content', '')
        print(f"内容: {content if content else '<空>'}")
        # 处理附加信息
        additional_kwargs = getattr(msg, 'additional_kwargs', {})
        if additional_kwargs:
            print("附加信息:")
            for key, value in additional_kwargs.items():
                if key == 'tool_calls' and value:
                    print("  工具调用:")
                    for tool_call in value:
                        print(f"    - ID: {tool_call['id']}")
                        print(f"      函数: {tool_call['function']['name']}")
                        print(f"      参数: {tool_call['function']['arguments']}")
                else:
                    print(f"  {key}: {value}")
        # 处理 ToolMessage 特有字段
        if msg_type == 'ToolMessage':
            tool_name = getattr(msg, 'name', '')
            tool_call_id = getattr(msg, 'tool_call_id', '')
            print(f"工具名称: {tool_name}")
            print(f"工具调用 ID: {tool_call_id}")
        # 处理 AIMessage 的工具调用和元数据
        if msg_type == 'AIMessage':
            tool_calls = getattr(msg, 'tool_calls', [])
            if tool_calls:
                print("工具调用:")
                for tool_call in tool_calls:
                    print(f"  - 名称: {tool_call['name']}")
                    print(f"    参数: {tool_call['args']}")
                    print(f"    ID: {tool_call['id']}")
            # 提取元数据
            metadata = getattr(msg, 'response_metadata', {})
            if metadata:
                print("元数据:")
                token_usage = metadata.get('token_usage', {})
                print(f"  令牌使用: {token_usage}")
                print(f"  模型名称: {metadata.get('model_name', '未知')}")
                print(f"  完成原因: {metadata.get('finish_reason', '未知')}")
        # 打印消息 ID
        msg_id = getattr(msg, 'id', '未知')
        print(f"消息 ID: {msg_id}")
        print("-" * 50)


# API接口:运行智能体并返回大模型结果或中断数据
@app.post("/agent/invoke", response_model=AgentResponse)
@traceable(run_type="chain", name="对话入口")
async def invoke_agent(request: AgentRequest):
    logger.info(f"调用/agent/invoke接口，运行智能体并返回大模型结果或中断数据，接受到前端用户请求:{request}")
    # 获取用户请求中的user_id和session_id
    user_id = request.user_id
    session_id = request.session_id


    # # 调用函数获取长期记忆
    # result = await read_long_term_info(user_id)
    # # 检查返回结果是否成功
    # if result.get("success", False):
    #     long_term_info = result.get("long_term_info")
    #     # 若获取到的内容不为空 则将记忆内容拼接到系统提示词中
    #     if long_term_info:
    #         system_message = f"{request.system_message}我的附加信息有:{long_term_info}"
    #         logger.info(f"获取用户偏好配置数据，system_message的信息为:{system_message}")
    #     # 若获取到的内容为空，则直接使用系统提示词
    #     else:
    #         system_message = request.system_message
    #         logger.info(f"未获取到用户偏好配置数据，system_message的信息为:{system_message}")
    # else:
    #     system_message = request.system_message
    #     logger.info(f"未获取到用户偏好配置数据，system_message的信息为:{system_message}")

    # 判断当前用户会话是否存在
    exists = await app.state.session_manager.session_id_exists(user_id, session_id)

    # 若用户会话不存在 则创建新会话
    if not exists:
        status = "idle"
        last_query = None
        last_response = None
        last_updated = time.time()
        ttl = Config.TTL
        # 创建会话并存储到redis中
        await app.state.session_manager.create_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

    # 新请求统一更新会话信息
    status = "running"
    last_query = request.query
    last_response = None
    last_updated = time.time()
    ttl = Config.TTL
    await app.state.session_manager.update_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

    # 构造智能体输入消息体
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "is_same_package_user_count": 0,
        "intent_recognize_count": 0,
        "matched_band_address_count": 0,
        "get_band_info_by_idcard_count": 0
    }
    try:
        # 先调用智能体
        result = await app.state.agent.ainvoke(initial_state, config={"configurable": {"thread_id": session_id}}, version="v2")
        # 将返回的messages进行格式化输出 方便查看调试
        #await parse_messages(result["messages"])

        # 再处理结果并更新会话状态
        return await process_agent_result(session_id, result, user_id)
    except GraphInterrupt as interrupt_ex:
        # 提取中断提示文本
        interrupt_hints = [interrupt.value for interrupt in interrupt_ex.interrupts]
        interrupt_msg = "\n".join(interrupt_hints)
        logger.info(f"智能体触发用户输入中断，提示内容: {interrupt_msg}")

        # 返回中断类型给前端，区分正常回答/中断/错误
        interrupt_response = AgentResponse(
            session_id=session_id,
            status="interrupt",  # 新增中断状态，前端识别需要用户输入
            message=interrupt_msg
        )
        # 更新会话状态为中断
        await app.state.session_manager.update_session(
            user_id, session_id,
            status="interrupt",
            last_query=request.query,
            last_response=interrupt_response,
            last_updated=time.time(),
            ttl=Config.TTL
        )
        return interrupt_response
    except Exception as e:
        # 异常处理
        error_response = AgentResponse(
            session_id=session_id,
            status="error",
            message=f"处理请求时出错: {str(e)}"
        )
        logger.error(f"处理请求时出错: {error_response}")

        # 更新会话状态
        status = "error"
        last_query = None
        last_response = error_response
        last_updated = time.time()
        ttl = Config.TTL
        await app.state.session_manager.update_session(user_id, session_id, status, last_query, last_response, last_updated, ttl)

        return error_response


# API接口:恢复被中断的智能体执行，等待执行完成或再次中断
@app.post("/agent/resume", response_model=AgentResponse)
@traceable(run_type="chain", name="中断后对话入口")
async def resume_agent(response: AgentRequest):
    logger.info(f"resume_agent接口，接受到前端用户请求:{response}")
    # 获取用户请求中的user_id和session_id
    user_id = response.user_id
    client_session_id = response.session_id

    # 判断当前用户会话是否存在
    exists = await app.state.session_manager.user_id_exists(user_id)
    # 若用户不存在 则抛出异常
    if not exists:
        logger.error(f"status_code=404,用户会话 {user_id} 不存在")
        raise HTTPException(status_code=404, detail=f"用户会话 {user_id} 不存在")

    # 然后判断会话ID是否匹配 若不匹配则抛出异常
    session = await app.state.session_manager.get_session(user_id, client_session_id)
    server_session_id = session.get("session_id")
    if server_session_id != client_session_id:
        logger.error(f"status_code=400,会话ID不匹配，可能是过期的请求")
        raise HTTPException(status_code=400, detail="会话ID不匹配，可能是过期的请求")

    # 检查会话状态是否为中断 若不是中断则抛出异常
    session = await app.state.session_manager.get_session(user_id, client_session_id)
    status = session.get("status")
    if status != "interrupted":
        logger.error(f"status_code=400,会话当前状态为 {status}，无法恢复非中断状态的会话")
        raise HTTPException(status_code=400, detail=f"会话当前状态为 {status}，无法恢复非中断状态的会话")

    # 更新会话状态
    status = "running"
    last_query = None
    last_response = None
    last_updated = time.time()
    await app.state.session_manager.update_session(user_id, status, last_query, last_response, last_updated)
    config = {"configurable": {"thread_id": server_session_id}}
    query = response.query
    try:

        await app.state.agent.aupdate_state(
            config,
            {"messages": [HumanMessage(query)]}
        )
        # 先恢复智能体执行
        result = await app.state.agent.ainvoke(Command(resume=query), config=config, version="v2")

        # 再处理结果并更新会话状态
        return await process_agent_result(server_session_id, result, user_id)
    except GraphInterrupt as interrupt_ex:
        # 提取中断提示文本
        interrupt_hints = [interrupt.value for interrupt in interrupt_ex.interrupts]
        interrupt_msg = "\n".join(interrupt_hints)
        logger.info(f"智能体触发用户输入中断，提示内容: {interrupt_msg}")

        # 返回中断类型给前端，区分正常回答/中断/错误
        interrupt_response = AgentResponse(
            session_id=server_session_id,
            status="interrupt",  # 新增中断状态，前端识别需要用户输入
            message=interrupt_msg
        )
        # 更新会话状态为中断
        await app.state.session_manager.update_session(
            user_id, server_session_id,
            status="interrupt",
            last_query=response.query,
            last_response=interrupt_response,
            last_updated=time.time(),
            ttl=Config.TTL
        )
        return interrupt_response
    except Exception as e:
        # 异常处理
        error_response = AgentResponse(
            session_id=server_session_id,
            status="error",
            message=f"恢复执行时出错: {str(e)}"
        )
        logger.error(f"处理请求时出错: {error_response}")

        # 更新会话状态
        status = "error"
        last_query = None
        last_response = error_response
        last_updated = time.time()
        await app.state.session_manager.update_session(user_id, status, last_query, last_response, last_updated)

        return error_response



# 启动服务器
if __name__ == "__main__":
    import uvicorn

    # 注意：reload=True 在某些 Windows 配置下可能需要额外处理，
    # 但设置上述策略通常能解决主要的 psycopg 报错
    uvicorn.run("agent_api:app", host="0.0.0.0", port=8012, reload=True)

   # uvicorn.run(app, host=Config.HOST, port=Config.PORT)
