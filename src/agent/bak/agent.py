from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
# 导入可运行配置类
from langchain_core.runnables import RunnableConfig
import threading
# 导入LangChain的提示模板类
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from typing import Annotated, Sequence
# 导入LangChain的消息基类
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
# 导入消息处理函数，用于追加消息
from langgraph.graph.message import add_messages
import sys
# 导入统一的 Config 类
from config import Config
from llms import get_llm
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

class MessagesState(TypedDict):
    # 定义messages字段，类型为消息序列，使用add_messages处理追加
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_broadband_related: str        #是否宽带问题
    fault_type: str                  #故障类型
    emotion_code: str                #用户情绪
    intention_code: str              #用户宽带问题诉求
    step: str                        #当前步骤

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

    try:
        # 先检查缓存，无锁访问
        if template_file in create_chain.prompt_cache:
            prompt_template = create_chain.prompt_cache[template_file]
            logger.info(f"Using cached prompt template for {template_file}")
        else:
            # 使用锁保护缓存访问
            with create_chain.lock:
                # 检查缓存中是否已有该模板
                if template_file not in create_chain.prompt_cache:
                    BASE_DIR = Path(__file__).parent.parent
                    logger.info(f"Loading and caching prompt template from {template_file}")
                    file_path = BASE_DIR / template_file
                    # 直接加载，不搞复杂缓存，避免死锁
                    prompt_template = PromptTemplate.from_file(file_path, encoding="utf-8")
                    # 从文件加载提示模板并存入缓存
                    create_chain.prompt_cache[file_path] = PromptTemplate.from_file(file_path, encoding="utf-8")
                # 从缓存中获取提示模板
                prompt_template = create_chain.prompt_cache[file_path]

        # 创建聊天提示模板，使用模板内容
        prompt = ChatPromptTemplate.from_messages([("human", prompt_template.template)])
        # 返回提示模板与LLM的组合链，若有结构化输出则绑定
        return prompt | (llm_chat.with_structured_output(structured_output) if structured_output else llm_chat)
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise


#开始节点，询问用户宽带遇到什么问题了吗？
def start_node(state: MessagesState):
    return {
        "messages": [AIMessage(content="您好，请问您的宽带遇到什么问题了吗？")],
        "is_broadband_related": "",
        "fault_type": "",
        "emotion_code": "",
        "intention_code": "",
        "step": "start"
    }


#意图识别（确认用户是要解决宽带问题，及识别故障类型、当前情绪）
def intent_recognize_node(state: MessagesState, config: RunnableConfig, llm_chat):
    # 记录代理开始处理查询
    logger.info("意图识别中...")
    try:
        # 自定义线程内存储逻辑 过滤消息
        messages = filter_messages(state["messages"])
        last_user_msg = messages[-1].content if messages else ""
        # 创建代理处理链
        agent_chain = create_chain(llm_chat, Config.PROMPT_INTENT_RECOGNIZE)
        # 调用代理链处理消息
        response = agent_chain.invoke({"question": last_user_msg, "messages": [m.content for m in messages]})
        # 返回更新后的对话状态
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        # 记录错误日志
        logger.error(f"意图识别错误: {e}")
        # 返回错误消息
        return {"messages": [AIMessage(content=f"处理请求时出错: {e}")]}

def create_graph(llm_chat)-> StateGraph:
    #创建状态图实例，使用MessagesState作为状态类型
    workflow = StateGraph(MessagesState)
    workflow.add_node("start_node",start_node)
    workflow.add_node("intent_recognize_node", lambda state, config: intent_recognize_node(state, config, llm_chat=llm_chat))

    # 添加从起始到代理的边
    workflow.add_edge(START, "start_node")
    workflow.add_edge("start_node", "intent_recognize_node")
    workflow.add_edge("intent_recognize_node", END)
    # 编译状态图，绑定检查点和存储
    return workflow.compile()



# 定义响应函数
def graph_response(graph: StateGraph, user_input: str, config: dict) -> None:
    """处理用户输入并输出响应，区分工具输出和大模型输出，支持多工具。

    Args:
        graph: 状态图实例。
        user_input: 用户输入。
        config: 运行时配置。
    """
    try:
        # 启动状态图流处理用户输入
        events = graph.stream({"messages": [{"role": "user", "content": user_input}]}, config)
        # 遍历事件流
        for event in events:
            # 遍历事件中的值
            for value in event.values():
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

        welcome = get_ai_response(graph, {"messages": []}, config)
        print(f"Assistant: {welcome}")
        # 进入主循环
        while True:

            # 获取用户输入并去除首尾空格
            user_input = input("User: ").strip()
            # 检查是否退出
            if user_input.lower() in {"quit", "exit", "q"}:
                print("拜拜!")
                break
            # 检查输入是否为空
            if not user_input:
                print("请输入聊天内容！")
                continue
            # 处理用户输入并选择是否流式输出响应
            graph_response(graph, user_input, config)
    except Exception as e:
        logger.error(f"Graph creation failed: {e}")
        print(f"错误: {e}")
        sys.exit(1)

# 检查是否为主模块运行
if __name__ == "__main__":
    # 调用主函数
    main()







