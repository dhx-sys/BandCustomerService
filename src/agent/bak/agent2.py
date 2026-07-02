from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.errors import GraphInterrupt
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
# 导入可运行配置类
from langchain_core.runnables import RunnableConfig
import threading
# 导入LangChain的提示模板类
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from typing import Annotated, Sequence, Literal
# 导入LangChain的消息基类
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
# 导入消息处理函数，用于追加消息
from langgraph.graph.message import add_messages
import sys
# 导入统一的 Config 类
from config import Config
from llms import get_llm
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

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

class MessagesState(TypedDict):
    # 定义messages字段，类型为消息序列使用add_messages处理追加，
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_broadband: bool         #是否宽带问题
    fault_type: str                  #故障类型
    Judgment_result: str           #是否同套餐
    ususer_score: bool             #是否本网用户
    intention_code: str
    say_to_user: str
    reason: str
    is_same_package_user_count: int #是否同套餐循环控制
    intent_recognize_count: int  # 是否解决宽带问题循环控制

class IntentRecognizeResult(BaseModel):
    model_config = ConfigDict(strict=False)
    is_broadband: bool = Field(description="是否为宽带问题")
    fault_type: str = Field(default="", description="宽带故障类型")
    intention_code: str = Field(default="", description="用户意图")
    say_to_user: str = Field(description="继续和用户说的话")
    reason: str = Field(description="判断理由，用于调试")

class IsSamePackageResult(BaseModel):
    model_config = ConfigDict(strict=False)
    Judgment_result: str = Field(default="nuknown", description="模型返回的结果:yes/no/nuknown")
    say_to_user: str = Field(description="继续和用户说的话")
    reason: str = Field(description="判断理由，用于调试")
# 定义线程内的持久化存储消息过滤函数
def filter_messages(messages: list) -> list:
    """过滤消息列表，仅保留 AIMessage 和 HumanMessage 类型消息"""
    # 过滤出 AIMessage 和 HumanMessage 类型的消息
    filtered = [msg for msg in messages if msg.__class__.__name__ in ['AIMessage', 'HumanMessage']]
    # 如果过滤后的消息超过N条，返回最后N条，否则返回过滤后的完整列表
    return filtered[-5:] if len(filtered) > 5 else filtered

# 定义创建处理链的函数
def create_chain(llm_chat, template_file: str, structured_output=None):
    """创建 LLM 处理链，加载提示模板并绑定模型，使用缓存避免重复读取文件。

    Args:
        llm_chat: 语言模型实例。
        template_file: 提示模板文件路径。
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

#nodes
#意图识别（确认用户是要解决宽带问题，及识别故障类型）
def intent_recognize_node(state: MessagesState, config: RunnableConfig, llm_chat):
    # 记录代理开始处理查询
    logger.info("意图识别中...")
    try:
        user_input = ""
        old_count = state["intent_recognize_count"]
        if old_count > 0:
            user_input = input("userI: ")
        # 自定义线程内存储逻辑 过滤消息
        messages = filter_messages(state["messages"])
        #进线后由机器人先询问用户，判断如果历史消息中没有用户消息
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_INTENT_RECOGNIZE,IntentRecognizeResult)
        # 调用代理链处理消息
        response: IntentRecognizeResult = agent_chain.invoke({"question": user_input, "messages": [m.content for m in messages]})
        # 返回更新后的对话状态
        print(response)
        return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)], "fault_type": response.fault_type, "intent_recognize_count": old_count+1}
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


#接口查询是否本网用户
def is_us_user(state: MessagesState):
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
        is_us_user = api_data["data"]["is_us_user"]

        return {
            # 保持消息不变
            "messages": state["messages"],
            "ususer_score": is_us_user
        }

    except Exception as e:
        # API失败时的处理
        result = f"API调用失败：{str(e)}"
        # 返回更新后的状态
        return {
            # 保持消息不变
            "messages": state["messages"],
            "ususer_score": False
        }


#是否同套餐下的用户
def is_same_package_user(state: MessagesState, config: RunnableConfig, llm_chat):
    #请问是来电号码底下的宽带吗？
    logger.info("是否为同套餐下的宽带识别中。。。")
    try:
        user_input = ""
        old_count = state["is_same_package_user_count"]
        if old_count > 0:
            user_input = input("userI: ")
            # 触发 LangGraph 中断，对外抛出中断信息（给到前端/对话层）
            #interrupt("")
        # 自定义线程内存储逻辑 过滤消息
        messages = filter_messages(state["messages"])
       # last_user_msg = messages[-1].content if isinstance(messages[-1], HumanMessage) else "-"
        #取最新一条用户消息
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_IS_SAME_PACKAGE_USER, IsSamePackageResult)
        # 调用代理链处理消息
        response: IsSamePackageResult = agent_chain.invoke(
            {"question": user_input, "messages": [m.content for m in messages]})
        # 返回更新后的对话状态
        print(response)
        return {"messages": [HumanMessage(content=user_input), AIMessage(content=response.say_to_user)], "Judgment_result": response.Judgment_result, "is_same_package_user_count": old_count+1}
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
# 新增节点：user_id_car_get_info
def user_id_car_get_info(state: MessagesState):
    """兜底节点，no 分支进入，示例逻辑"""
    return {
        "messages": [AIMessage(content="已为您进入其他业务流程")]
    }

#router
#根据用户回答是否解决宽带问题的结果决定一下步路由
def router_after_intent_recognize_node(state: MessagesState) -> Literal["is_us_user", "intent_recognize_node"]:
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
    isBroadband = state.get("is_broadband")
    logger.info(f"Routing based on relevance_score: {faultType}")
    if isBroadband:  #判断如果是解决宽带问题
        if faultType:  #如果已经识别到宽带问题
            return "is_us_user"
        else:
            return "intent_recognize_node"  #没有识别到继续循环
    else:
        return END  #不是解决宽带问题退出


#根据用户查询结果用户是否本网用户结果决定下一步路由
def route_after_is_us_user(state: MessagesState) -> Literal["is_same_package_user", "user_id_car_get_info"]:
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到 rewrite
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to rewrite")
        return "user_id_car_get_info"
    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to rewrite")
        return "user_id_car_get_info"
        # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to rewrite")
        return "user_id_car_get_info"
    # 获取状态中的 relevance_score，若不存在则返回 None
    ususer_score = state.get("ususer_score")

    logger.info(f"Routing based on relevance_score: {ususer_score}")
    if ususer_score:
        return "is_same_package_user"
    else:
        return "user_id_car_get_info"


#根据用户回答是否同套餐下的宽带结果决定一下步路由
def router_after_is_same_package_user(state: MessagesState) -> Literal["is_same_package_user", "user_id_car_get_info"]:
    # 检查状态是否为有效字典，若无效则记录错误并默认路由到 rewrite
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to intent_recognize_node")
        return "is_same_package_user"
    # 检查状态是否包含 messages 字段，若缺失则记录错误并默认路由到 rewrite
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to intent_recognize_node")
        return "is_same_package_user"
        # 检查 messages 是否为空，若为空则记录警告并默认路由到 rewrite
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to intent_recognize_node")
        return "is_same_package_user"
        # 获取状态中的 relevance_score，若不存在则返回 None
    Judgment_result = state.get("Judgment_result")
    logger.info(f"Routing based on relevance_score: {Judgment_result}")
    if Judgment_result == 'no':
        return "user_id_car_get_info"
    elif Judgment_result == 'yes':
        return END
    else:
        return "is_same_package_user"

def create_graph(llm_chat)-> StateGraph:
    #创建状态图实例，使用MessagesState作为状态类型
    workflow = StateGraph(MessagesState)
    workflow.add_node("intent_recognize_node", lambda state, config: intent_recognize_node(state, config, llm_chat=llm_chat))
    workflow.add_node("is_us_user", lambda state, config: is_us_user(state))
    workflow.add_node("is_same_package_user", lambda state, config: is_same_package_user(state, config, llm_chat=llm_chat))
    workflow.add_node("user_id_car_get_info",
                      lambda state, config: user_id_car_get_info(state, config, llm_chat=llm_chat))
    # 添加从起始到代理的边
    workflow.add_edge(START, "intent_recognize_node")
    workflow.add_conditional_edges("intent_recognize_node", lambda state: router_after_intent_recognize_node(state))
    workflow.add_conditional_edges("is_us_user", lambda state: route_after_is_us_user(state))
    workflow.add_conditional_edges("is_same_package_user", lambda state: router_after_is_same_package_user(state))
    workflow.add_edge("user_id_car_get_info", END)
    # 编译状态图，绑定检查点和存储
    return workflow.compile()



# 定义响应函数
def graph_response(graph: StateGraph, config: dict) -> None:
    """处理用户输入并输出响应，区分工具输出和大模型输出，支持多工具。

    Args:
        graph: 状态图实例。
        user_input: 用户输入。
        config: 运行时配置。
    """
    try:
        initial_state = {
            "messages": [],
            "is_same_package_user_count": 0,
            "intent_recognize_count": 0,
            "is_broadband_related": "",
            "fault_type": "",
            "emotion_code": "",
            "intention_code": "",
            "step": ""
        }

        # 启动状态图流处理用户输入
        events = graph.stream(initial_state, config)
        # 遍历事件流
        for event in events:
            # 遍历事件中的值
            for node_name, value in event.items():
                if node_name in ["is_us_user"]:
                    continue

                # 检查是否有有效消息
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("No valid messages in response")
                    continue

                # 获取最后一条消息
                last_message = value["messages"][-1]

                # 检查消息是否有内容
                if hasattr(last_message, "content"):
                    content = last_message.content
                    print(f"Assistant: {content}")
                else:
                    # 如果消息没有内容，可能是中间状态
                    logger.info("Message has no content, skipping")
                    print("Assistant: 未获取到相关回复")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
        print("Assistant: 处理响应时发生值错误")
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        print("Assistant: 处理响应时发生未知错误")

def get_ai_response(graph, input_state, config):
    final_state = graph.invoke(input_state, config)
    return final_state["messages"][-1].content
def main():
    try:
        # 调用get_llm函数初始化Chat模型实例和Embedding模型实例
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        graph = create_graph(llm_chat)
        # 打印机器人就绪提示
        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
        # 定义运行时配置，包含线程ID和用户ID
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}
        # 进入主循环
        while True:
            # 处理用户输入并选择是否流式输出响应
            graph_response(graph, config)
    except GraphInterrupt as e:
        # 捕获节点内 interrupt 抛出的中断，取出提问话术
        interrupt_tip = e.value
        logger.info(f"流程中断，等待用户回复: {interrupt_tip}")
        return interrupt_tip
    except Exception as e:
        logger.error(f"Graph creation failed: {e}")
        print(f"错误: {e}")
        sys.exit(1)

# 检查是否为主模块运行
if __name__ == "__main__":
    # 调用主函数
    main()







