from langchain_core.tools import tool
from langgraph.types import interrupt
from state import MessagesState, IntentRecognizeResult, checkIdCardResult, IsSamePackageResult,MatchedBandAddresSResult
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
import logging
from config import Config
from typing import Literal
from concurrent_log_handler import ConcurrentRotatingFileHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import threading
from pathlib import Path
from langchain.agents import create_agent
# # 设置日志基本配置，级别为DEBUG或INFO
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
# 定义线程内的持久化存储消息过滤函数
def filter_messages(messages: list) -> list:
    """过滤消息列表，仅保留 AIMessage 和 HumanMessage 类型消息"""
    # 过滤出 AIMessage 和 HumanMessage 类型的消息
    filtered = [msg for msg in messages if msg.__class__.__name__ in ['AIMessage', 'HumanMessage']]
    # 如果过滤后的消息超过N条，返回最后N条，否则返回过滤后的完整列表
    #return filtered[-5:] if len(filtered) > 5 else filtered
    return filtered
# 定义创建处理链的函数
def create_chain(llm_chat, template_file: str, structured_output=None):
    """创建 LLM 处理链，加载提示模板并绑定模型，使用缓存避免重复读取文件。

    Args:
        llm_chat: 语言模型实例。
        template_file: 提示词模板文件路径。
        structured_output: 可选的结构化输出模型。

    Returns:
        Runnable: 配置好的处理链。

    Raises:
        FileNotFoundError: 如果模板文件不存在。
    """
    # 定义静态缓存和锁（仅在函数第一次调用时初始化）
    if not hasattr(create_chain, "prompt_cache"):
        # 缓存字典
        create_chain.prompt_cache = {}
        # 线程锁 确保缓存的读写是线程安全的
        create_chain.lock = threading.Lock()
    BASE_DIR = Path(__file__).parent.parent
    file_path = BASE_DIR / template_file
    try:
        # 先检查缓存，无锁访问
        if file_path in create_chain.prompt_cache:
            prompt_template = create_chain.prompt_cache[file_path]
            logger.info(f"Using cached prompt template for {file_path}")
        else:
            # 使用锁保护缓存访问
            with create_chain.lock:
                # 检查缓存中是否已有该模板
                if file_path not in create_chain.prompt_cache:

                    logger.info(f"Loading and caching prompt template from {file_path}")

                    # 从文件加载提示模板并存入缓存
                    create_chain.prompt_cache[file_path] = PromptTemplate.from_file(file_path, encoding="utf-8")
                # 从缓存中获取提示模板
                prompt_template = create_chain.prompt_cache[file_path]

        # 创建聊天提示模板，使用模板内容
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template.template),
            ("human", "用户问题：{question}\n上下文：{messages}")
        ])
        # 返回提示模板与LLM的组合链，若有结构化输出则绑定
        if structured_output:
            llm = llm_chat.with_structured_output(structured_output)
        else:
            llm = llm_chat

        return prompt | llm
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise
#获取提示词模板
def get_prompt(template_file: str):
    """创建 LLM 处理链，加载提示模板并绑定模型，使用缓存避免重复读取文件。

    Args:
        llm_chat: 语言模型实例。
        template_file: 提示词模板文件路径。
        structured_output: 可选的结构化输出模型。

    Returns:
        Runnable: 配置好的处理链。

    Raises:
        FileNotFoundError: 如果模板文件不存在。
    """
    # 定义静态缓存和锁（仅在函数第一次调用时初始化）
    if not hasattr(create_chain, "prompt_cache"):
        # 缓存字典
        create_chain.prompt_cache = {}
        # 线程锁 确保缓存的读写是线程安全的
        create_chain.lock = threading.Lock()
    BASE_DIR = Path(__file__).parent.parent
    file_path = BASE_DIR / template_file
    try:
        # 先检查缓存，无锁访问
        if file_path in create_chain.prompt_cache:
            prompt_template = create_chain.prompt_cache[file_path]
            logger.info(f"Using cached prompt template for {file_path}")
        else:
            # 使用锁保护缓存访问
            with create_chain.lock:
                # 检查缓存中是否已有该模板
                if file_path not in create_chain.prompt_cache:

                    logger.info(f"Loading and caching prompt template from {file_path}")

                    # 从文件加载提示模板并存入缓存
                    create_chain.prompt_cache[file_path] = PromptTemplate.from_file(file_path, encoding="utf-8")
                # 从缓存中获取提示模板
                prompt_template = create_chain.prompt_cache[file_path]

        # 创建聊天提示模板，使用模板内容
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template.template),
            ("human", "用户问题：{question}\n上下文：{messages}")
        ])
        return prompt
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise


@tool
def human_input(hint: str) -> str:
    """向用户发起提问，获取用户回复，自动触发中断"""
    return interrupt({
        "interrupt_type": "user_ask",
        "content": hint
    })

#nodes
#意图识别（确认用户是要解决宽带问题，及识别故障类型）
# def intent_recognize_node(state: MessagesState, config: RunnableConfig, llm_chat):
#     # 记录代理开始处理查询
#     logger.info("意图识别开始...")
#     messages = filter_messages(state["messages"])
#     # 从config里取出全局统一的checkpointer，
#     react_prompt = get_prompt(Config.PROMPT_INTENT_RECOGNIZE, IntentRecognizeResult)
#     #进线后由机器人先询问用户，判断如果历史消息中没有用户消息
#     agent = create_agent(
#         model=llm_chat,
#         tools=[human_input],
#         checkpointer=Config.GLOBAL_CHECKPOINTER,
#         system_prompt=react_prompt
#     )
#     agent_res = agent.invoke({"question": messages[-1], "messages": messages}, config=config)
#     try:
#         # 获取最后一条AI消息
#         last_ai_msg: AIMessage = agent_res["messages"][-1]
#         content_data = json.loads(last_ai_msg.content)
#         return {
#             "messages": [AIMessage(content=content_data["say_to_user"])],
#             "fault_type": content_data["fault_type"]
#                 }
#     except Exception as e:
#         # 记录错误日志
#         logger.error(f"意图识别错误: {e}")
#         # 返回错误消息
#         import traceback
#         print(f"\n🔴 节点执行崩溃！异常类型: {type(e).__name__}")
#         traceback.print_exc()
#         return {
#             "messages": [AIMessage(content="系统处理出错，请稍后再试。")]
#         }

#接口查询是否本网用户
def is_us_user():
    try:
        # 替换成你真实的API地址、参数、headers
        # response = requests.get(
        #     url="https://jsonplaceholder.typicode.com/todos/1",
        #     timeout=10  # 超时控制
        # )
        # response.raise_for_status()
        # api_data = response.json()
        api_data = {
            "code": 200,
            "status": "success",
            "data": {
                "user_id": 1001,
                "name": "测试用户",
                "tel": "19909466205",
                "is_us_user": True,
                "result": "已获取外部系统信息"
            },
            "message": "模拟API调用成功"
        }
        return api_data["data"]["is_us_user"]

    except Exception as e:
        # API失败时的处理
        logger.error(f"查询是否本网用户失败: {e}")
        return False

#来电号码查询宽带信息
def get_band_info_by_phone():
    try:
        api_data = {
            "code": 200,
            "status": "success",
            "data": {
                "user_id": 1001,
                "name": "测试用户",
                "tel": "19909466205",
                "band_id": "1005213",
                "result": "已获取外部系统信息",
                "band_info": ['甘肃省兰州市城关区阳光花园小区 5 栋 3 单元 201 室',
                              '甘肃省兰州市城关区渭源路街道南昌路社区兰宁小区3号楼2单元7层702',
                              '甘肃省天水市秦州区杨庄村 2 社 18 号']

            },
            "message": "模拟API调用成功"
        }
        return {
            "band_id":api_data["data"]["band_id"],
            "band_info":api_data["data"]["band_info"]
        }

    except Exception as e:
        # API失败时的处理
        logger.error(f"电话号码查询宽带信息失败: {e}")
        return ""

#身份证号码查询宽带信息
def get_band_info_by_idcard(state: MessagesState,config: RunnableConfig,  llm_chat):
    try:
        user_input = ""
        old_count = state["get_band_info_by_idcard_count"]
        if old_count > 0:
            user_input = input("userI: ")
            # 自定义线程内存储逻辑 过滤消息
            messages = filter_messages(state["messages"])
            # 进线后由机器人先询问用户，判断如果历史消息中没有用户消息
            # 创建代理处理链
            agent_chain = create_chain(llm_chat, Config.PROMPT_CHECK_ID_CARD, checkIdCardResult)
            # 调用代理链处理消息
            response: IntentRecognizeResult = agent_chain.invoke(
                {"question": user_input, "messages": [m.content for m in messages]})
            # 返回更新后的对话状态
            print(response)
            if response.check_result: #身份证号码校验通过，按照身份证号码查询宽带信息
                api_data = {
                    "code": 200,
                    "status": "success",
                    "data": {
                        "user_id": 1001,
                        "name": "测试用户",
                        "tel": "19909466205",
                        "band_id": "1005213",
                        "result": "已获取外部系统信息",
                        "band_info": ['甘肃省兰州市城关区阳光花园小区 5 栋 3 单元 201 室',
                                      '甘肃省兰州市城关区渭源路街道南昌路社区兰宁小区3号楼2单元7层702',
                                      '甘肃省天水市秦州区杨庄村 2 社 18 号']
                    },
                    "message": "模拟API调用成功"
                }
                if api_data["data"]["band_id"]: #如果查询到宽带信息
                    return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)],
                            "check_result": response.check_result,
                            "get_band_info_by_idcard_count": old_count + 1,
                            "get_band_info": True,
                            "band_info": api_data["data"]["band_info"]
                            }
                else:
                    return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)],
                            "check_result": response.check_result,
                            "get_band_info_by_idcard_count": old_count + 1,
                            "get_band_info": False
                            }
            else:
                return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)],
                        "check_result": response.check_result,
                        "get_band_info_by_idcard_count": old_count + 1,
                        "get_band_info": False
                        }
    except Exception as e:
        # API失败时的处理
        logger.error(f"身份证号码查询宽带信息失败: {e}")
        return ""

def get_user_info(state: MessagesState, config: RunnableConfig, llm_chat):
    # 记录代理开始处理查询
    logger.info("信息收集开始...")
    try:
        ususer_score = is_us_user()
        if not ususer_score:  #如果不是同网用户，
            return{
                "need_id_card_get_info": True
            }
        # 继续收集是否套餐下的宽带
        logger.info("是否为同套餐下的宽带查询开始。。。")
        user_input = ""
        old_count = state["is_same_package_user_count"]
        if old_count > 0:
            user_input = input("userI: ")
            # 触发 LangGraph 中断，对外抛出中断信息（给到前端/对话层）
            # interrupt("")
        # 自定义线程内存储逻辑 过滤消息
        messages = filter_messages(state["messages"])
        # last_user_msg = messages[-1].content if isinstance(messages[-1], HumanMessage) else "-"
        # 取最新一条用户消息
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_IS_SAME_PACKAGE_USER, IsSamePackageResult)
        # 调用代理链处理消息
        response: IsSamePackageResult = agent_chain.invoke(
            {"question": user_input, "messages": [m.content for m in messages]})
        # 返回更新后的对话状态
        print(response)
        if response.same_package == "no":  #如果不是同套餐下的宽带，按照身份证号码进行查询
            return {
                "messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)],
                "same_package": response.same_package,
                "is_same_package_user_count": old_count + 1,
                "need_id_card_get_info": True
            }
        elif response.same_package == "yse" or response.same_package == "unknown":
            result = get_band_info_by_phone()
            if result["id_card"]:  # 如果来电号码查询到宽带信息，
                return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)],
                        "same_package": response.same_package,
                        "is_same_package_user_count": old_count + 1,
                        "need_id_card_get_info": False,
                        "band_info": result["band_info"]
                        }
            return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)],
                    "same_package": response.same_package,
                    "is_same_package_user_count": old_count + 1,
                    "need_id_card_get_info": True
                    }
        else:
            return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)],
                    "same_package": response.same_package,
                    "is_same_package_user_count": old_count + 1,
                    "need_id_card_get_info": False
                    }
    except Exception as e:
        # 记录错误日志
        logger.error(f"信息收集错误: {e}")
        # 返回错误消息
        import traceback
        print(f"\n🔴 节点执行崩溃！异常类型: {type(e).__name__}")
        traceback.print_exc()
        return {
            "messages": [AIMessage(content="系统处理出错，请稍后再试。")]
        }

#宽带信息校验
def check_band_info(state: MessagesState, config: RunnableConfig, llm_chat):
    # 记录代理开始处理查询
    logger.info("意图识别开始...")
    try:
        user_input = ""
        old_count = state["matched_band_address_count"]
        if old_count > 0:
            user_input = input("userI: ")
        # 自定义线程内存储逻辑 过滤消息
        messages = filter_messages(state["messages"])
        # 进线后由机器人先询问用户，判断如果历史消息中没有用户消息
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_MATCHED_BAND_ADDRESS, MatchedBandAddresSResult)
        # 调用代理链处理消息
        response: IntentRecognizeResult = agent_chain.invoke(
            {"question": user_input, "messages": [m.content for m in messages], "broadbandAddress": state["band_info"]})
        # 返回更新后的对话状态
        return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)],
                "check_address_result": response.check_address_result, "matched_band_address_count": old_count + 1}
    except Exception as e:
        # 记录错误日志
        logger.error(f"意图识别错误: {e}")
        # 返回错误消息
        import traceback
        print(f"\n🔴 节点执行崩溃！异常类型: {type(e).__name__}")
        traceback.print_exc()
        return {
            "messages": [AIMessage(content="系统处理出错，请稍后再试。")]
        }
#router
#根据用户回答是否解决宽带问题的结果决定一下步路由
def router_after_intent_recognize_node(state: MessagesState) -> Literal["get_user_info", "intent_recognize_node"]:
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到 rewrite
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to intent_recognize_node")
        return "intent_recognize_node"
    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to intent_recognize_node")
        return "intent_recognize_node"
        # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to intent_recognize_node")
        return "intent_recognize_node"
        # 获取状态中的 relevance_score，若不存在则返回 None
    faultType = state.get("fault_type")
    logger.info(f"Routing based on relevance_score: {faultType}")
    if faultType == 'M' or faultType == '':  #如果未识别到宽带问题继续执行该节点
        return "intent_recognize_node"
    else:
        return "get_user_info"  #识别到用户意图进行信息收集节点
#信息收集路由
def router_after_get_user_info_node(state: MessagesState) -> Literal["get_user_info", "get_band_info_by_idcard", "check_band_info"]:
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到 rewrite
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to get_user_info")
        return "get_user_info"
    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to get_user_info")
        return "get_user_info"
        # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to get_user_info")
        return "get_user_info"
        # 获取状态中的 relevance_score，若不存在则返回 None
    needIdCard = state.get("need_id_card_get_info")
    getBandInfo = state.get("get_band_info")
    logger.info(f"Routing based on relevance_score: {needIdCard}")
    if needIdCard:  #如果未识别到宽带问题继续执行该节点
        return "get_band_info_by_idcard"
    elif getBandInfo:
        return "check_band_info"
    else:
        return "get_user_info"  #识别到用户意图进行信息收集节点
#身份证号码查询宽带信息路由
def router_after_get_band_info_by_idcard(state: MessagesState) -> Literal["get_band_info_by_idcard", "check_band_info"]:
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到 rewrite
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to get_band_info_by_idcard")
        return "get_band_info_by_idcard"
    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to get_band_info_by_idcard")
        return "get_band_info_by_idcard"
        # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to get_band_info_by_idcard")
        return "get_band_info_by_idcard"
        # 获取状态中的 relevance_score，若不存在则返回 None
    getBandInfo = state.get("get_band_info")
    logger.info(f"Routing based on relevance_score: {getBandInfo}")
    if getBandInfo:
        return "check_band_info"
    else:
        return "get_band_info_by_idcard"  #识别到用户意图进行信息收集节点
#地址匹配路由
def router_after_check_band_info(state: MessagesState) -> Literal["check_band_info", END]:
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到 rewrite
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to check_band_info")
        return "check_band_info"
    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to check_band_info")
        return "check_band_info"
        # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to check_band_info")
        return "check_band_info"
        # 获取状态中的 relevance_score，若不存在则返回 None
    checkResult = state.get("check_address_result")
    logger.info(f"Routing based on relevance_score: {checkResult}")
    if checkResult == 'yes':  #如果未识别到宽带问题继续执行该节点
        return END
    else:
        return "check_band_info"  #识别到用户意图进行信息收集节点
