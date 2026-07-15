import json
import random
from uuid import UUID

from langgraph.types import interrupt
from langsmith import traceable, get_current_run_tree

from src.agent.state import MessagesState, IntentRecognizeResult, IsSamePackageResult, MatchedBandAddresSResult, \
    getFaultFodeResult, checkIdCardResult, OrdersInfo
from src.rag.ragIndex import query_knowledge
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
            ("human", "用户问题：{question}\n")
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


def get_prompt(template_file: str, structured_output=None):
    try:

        BASE_DIR = Path(__file__).parent.parent
        file_path = BASE_DIR / template_file
        with open(file_path, "r", encoding="utf-8") as f:
            raw_system_text = f.read()

        schema = structured_output.model_json_schema()
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

        full_system_template = raw_system_text.format(structured_schema=schema_text)
        return full_system_template
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise

#----------------------nodes------------------------------------------

#意图识别（确认用户是要解决宽带问题，及识别故障类型）
@traceable(run_type="chain", name="意图识别，引导用户说出宽带遇到的问题并分析用户意图")
async def intent_recognize_node(state: MessagesState, config: RunnableConfig, llm_chat):
    # 记录代理开始处理查询
    logger.info("意图识别开始...")
    #messages = filter_messages(state["messages"])
    query = state["messages"][-1].content
    run = get_current_run_tree()
    while True:
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_INTENT_RECOGNIZE, IntentRecognizeResult)
        # 调用代理链处理消息
        response: IntentRecognizeResult = await agent_chain.ainvoke({"question": query})
        if run:
            run.metadata["fault_type"] = response.fault_type
        assistant = {
            "interrupt_type": "user_input",
            "content": response.say_to_user,
            "payload": {}
        }
        interrupt(assistant)  # payload surfaces in result["__interrupt__"]
        faultType = response.fault_type
        if faultType:  # 如果识别到宽带问题继续执行该节点
            return {"messages": [AIMessage(content=response.say_to_user)], "fault_type": response.fault_type}


#接口查询是否本网用户
@traceable(run_type="chain", name="用户信息收集，查询接口判断是否本网用户")
async def is_us_user():
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
        #return ""

    except Exception as e:
        # API失败时的处理
        logger.error(f"查询是否本网用户失败: {e}")
        return False


#来电号码查询宽带信息
@traceable(run_type="chain", name="用户信息收集，调用接口查询来电号码下的宽带信息")
async def get_band_info(info):
    try:
        if info:
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
        else:
            return {
                "band_id": "",
                "band_info": ""
            }

    except Exception as e:
        # API失败时的处理
        logger.error(f"电话号码查询宽带信息失败: {e}")
        return ""


#身份证号码查询宽带信息
@traceable(run_type="chain", name="用户信息收集，调用接口查询身份证号码下的宽带信息")
async def get_band_info_by_idcard(state: MessagesState,config: RunnableConfig,  llm_chat):
        # 自定义线程内存储逻辑 过滤消息
        #messages = filter_messages(state["messages"])
        # 取最新一条用户消息
        query = state["messages"][-1].content if isinstance(state["messages"][-1], HumanMessage) else ""
        while True:
            # 创建代理处理链
            agent_chain = create_chain(llm_chat, Config.PROMPT_CHECK_ID_CARD, checkIdCardResult)
            # 调用代理链处理消息
            response: checkIdCardResult = await agent_chain.ainvoke({"question": query})
            assistant = {
                "interrupt_type": "user_input",
                "content": response.say_to_user,
                "payload": {}
            }
            interrupt(assistant)  # payload surfaces in result["__interrupt__"]
            if response.check_result: #身份证号码校验通过，按照身份证号码查询宽带信息
                result =await get_band_info(True)
                if result["band_id"]: #如果查询到宽带信息
                    return {"messages": [AIMessage(content=response.say_to_user)],
                            "check_result": response.check_result,
                            "get_band_info": True,
                            "band_info": result["band_info"]
                            }
                else:
                    assistant = {
                        "interrupt_type": "user_input",
                        "content": "您输入的身份证号码未查询到宽带信息，请重输",
                        "payload": {}
                    }

@traceable(run_type="chain", name="用户信息收集，引导用户说出是否同套餐下的宽带，并判断")
async def get_user_info(state: MessagesState, config: RunnableConfig, llm_chat):
    logger.info("信息收集开始...")
    ususer_score =await is_us_user()
    if not ususer_score:  #如果不是同网用户，
        return {
                "messages": [AIMessage(content="请输入身份证信息，便于查询宽带信息。")],
                "same_package": "请输入身份证信息，便于查询宽带信息。",
                "need_id_card_get_info": True
            }
    # 继续收集是否套餐下的宽带
    logger.info("是否为同套餐下的宽带查询开始。。。")
    # 自定义线程内存储逻辑 过滤消息
    #messages = filter_messages(state["messages"])
    # 取最新一条用户消息
    query = state["messages"][-1].content if isinstance(state["messages"][-1], HumanMessage) else ""
    while True:
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_IS_SAME_PACKAGE_USER, IsSamePackageResult)
        # 调用代理链处理消息
        response: IsSamePackageResult = await agent_chain.ainvoke({"question": query})
        assistant = {
            "interrupt_type": "user_input",
            "content": response.say_to_user,
            "payload": {}
        }
        interrupt(assistant)  # payload surfaces in result["__interrupt__"]
        if response.same_package == "no":  #如果不是同套餐下的宽带，按照身份证号码进行查询
            return {
                "messages": [AIMessage(content=response.say_to_user)],
                "same_package": response.same_package,
                "need_id_card_get_info": True
            }
        elif response.same_package == "yes" or response.same_package == "unknown":
            result =await get_band_info(True)
            if result["band_id"]:  # 如果来电号码查询到宽带信息，走验证地址环节
                return {"messages": [AIMessage(content=response.say_to_user)],
                        "same_package": response.same_package,
                        "need_id_card_get_info": False,
                        "band_info": result["band_info"],
                        "get_band_info": True
                        }
            return {"messages": [AIMessage(content="您来电号码未查询到宽带信息。")],
                    "same_package": response.same_package,
                    "need_id_card_get_info": True
                    }
        else:
            pass


#宽带信息校验
@traceable(run_type="chain", name="宽带地址信息校验")
async def check_band_info(state: MessagesState, config: RunnableConfig, llm_chat):
    # 记录代理开始处理查询
    logger.info("地址验证开始...")
    # 自定义线程内存储逻辑 过滤消息
    messages = filter_messages(state["messages"])
    #获取最后一条用户消息
    query = state["messages"][-1].content if isinstance(state["messages"][-1], HumanMessage) else ""
    while True:
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_MATCHED_BAND_ADDRESS, MatchedBandAddresSResult)
        # 调用代理链处理消息
        response: MatchedBandAddresSResult = await agent_chain.ainvoke({"question": query, "broadbandAddress": state["band_info"]})
        assistant = {
            "interrupt_type": "user_input",
            "content": response.say_to_user,
            "payload": {}
        }
        interrupt(assistant)  # payload surfaces in result["__interrupt__"]
        if response.check_address_result == 'yes':
            # 返回更新后的对话状态
            return {
                "messages": [AIMessage(content=response.say_to_user)],
                "address": response.address}


#解决宽带故障
@traceable(run_type="chain", name="解决用户宽带故障")
async def get_fault_code(state: MessagesState, config: RunnableConfig, llm_chat):
    #宽带故障一键查询（模拟：随机生成故障码）
    fault_code = random.choice(['100', '101', '10201', '10102', '103', '104', '105', '106', '201'])
    #知识库中查询对应故障处理流程
    manageWorkflow = query_knowledge(fault_code)

    messages = filter_messages(state["messages"])
    query = state["messages"][-1].content if isinstance(state["messages"][-1], HumanMessage) else ""
    while True:
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_FAULT_MANAGE, getFaultFodeResult)
        # 调用代理链处理消息
        response: getFaultFodeResult = await agent_chain.ainvoke(
            {"question": query, "messages": messages, "manageWorkflow": manageWorkflow})
        assistant = {
            "interrupt_type": "user_input",
            "content": response.say_to_user,
            "payload": {}
        }
        interrupt(assistant)  # payload surfaces in result["__interrupt__"]
        bandFault = response.band_fault
        if bandFault:  # 如果网络已恢复则结束workflow
            return {"messages": [AIMessage(content=response.say_to_user)],
                    fault_code: fault_code
                    }


#解决宽带故障
@traceable(run_type="chain", name="派单")
async def send_orders(state: MessagesState, config: RunnableConfig, llm_chat):
    order = OrdersInfo(
        id=UUID,
        thread_id=state["thread_id"],
        user_id=state["user_id"],
        band_id=state["band_id"],
        band_info=state["band_info"],
        band_address=state["address"],
        user_phone=state["user_phone"],
        fault_type=state["fault_type"],
        fault_code=state["fault_code"]
    )
    return {"messages": [AIMessage(content="已为您派单成功，24小时内，装维人员将与您进行联系，请注意接听电话，再见。")],
            "orders_info": order
            }





#------------------------router------------------------------------------

#信息收集路由
def router_after_get_user_info_node(state: MessagesState) -> Literal["get_band_info_by_idcard", "check_band_info"]:
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
    if needIdCard:
        return "get_band_info_by_idcard"
    if getBandInfo:
        return "check_band_info"


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


def router_after_get_fault_code(state: MessagesState) -> Literal["send_orders", END]:
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to get_fault_code")
        return "get_fault_code"
    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to get_fault_code")
        return "get_fault_code"
        # 检查 messages 是否为空，若为空则记录警告并默认路由到
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to get_fault_code")
        return "get_fault_code"
    faultTicket = state.get("fault_ticket")#是否需要派单
    logger.info(f"Routing based on relevance_score: {faultTicket}")
    if faultTicket:
        return "send_orders"
    else:
        return END