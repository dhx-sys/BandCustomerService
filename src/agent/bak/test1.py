import json
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool
from langgraph.types import interrupt
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Callable
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
import logging
from config import Config
from concurrent_log_handler import ConcurrentRotatingFileHandler
from pathlib import Path
from langchain.agents import create_agent

from llms import get_llm

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
    fault_type: str                  #用户意图
    ususer_score: bool             #是否本网用户
    intention_code: str
    say_to_user: str
    check_address_result: str    #地址是否匹配
    reason: str
    need_id_card_get_info: str   #是否需要身份证号码查询宽带信息
    is_same_package_user_count: int #是否同套餐循环控制
    intent_recognize_count: int  # 是否解决宽带问题循环控制
    get_band_info_by_idcard_count: int  # 身份证号码查询控制循环控制
    matched_band_address_count: int  #地址匹配循环控制
    get_band_info: bool   #是否获取到宽带信息
    band_info: str  #接口查询的宽带信息

class IntentRecognizeResult(BaseModel):
    model_config = ConfigDict(strict=False)
    fault_type: str = Field(description="用户意图")
    say_to_user: str = Field(description="下一轮给用户的回复")
    reason: str = Field(description="判断理由，用于调试")

def get_prompt(template_file: str, structured_output=None):
    try:

        BASE_DIR = Path(__file__).parent.parent.parent
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

@tool
def human_input(hint: str) -> str:
    """向用户发起提问，获取用户回复，自动触发中断"""
    print(hint)
    return interrupt({
        "interrupt_type": "user_ask",
        "content": hint
    })



def add_human_in_the_loop(
        tool: Callable | BaseTool,
        *,
        interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""

    # 检查传入的工具是否为 BaseTool 的实例
    if not isinstance(tool, BaseTool):
        # 如果不是 BaseTool，则将可调用对象转换为 BaseTool 对象
        tool = create_tool(tool)

    # 检查是否提供了 interrupt_config 参数
    if interrupt_config is None:
        # 如果未提供，则设置默认的人工中断配置，允许接受、编辑和响应
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    # 使用 create_tool 装饰器定义一个新的工具函数，继承原工具的名称、描述和参数模式
    @create_tool(
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )

    # 定义内部函数，用于处理带有中断逻辑的工具调用
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        # 创建一个人为中断请求，包含工具名称、输入参数和配置
        request: interrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call"
        }
        # 调用 interrupt 函数，获取人工审查的响应（取第一个响应）
        response = interrupt([request])[0]
        # 检查响应类型是否为“接受”（accept）
        if response["type"] == "accept":
            # 如果接受，直接调用原始工具并传入输入参数和配置
            tool_response = tool.invoke(tool_input, config)
        # 检查响应类型是否为“编辑”（edit）
        elif response["type"] == "edit":
            # 如果是编辑，更新工具输入参数为响应中提供的参数
            tool_input = response["args"]["args"]
            # 使用更新后的参数调用原始工具
            tool_response = tool.invoke(tool_input, config)
        # 检查响应类型是否为“响应”（response）
        elif response["type"] == "response":
            # 如果是响应，直接将用户反馈作为工具的响应
            user_feedback = response["args"]
            tool_response = user_feedback
        # 如果响应类型不被支持，则抛出异常
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        # 返回工具的响应结果
        return tool_response

    # 返回包装后的工具函数
    return call_tool_with_interrupt
#nodes
#意图识别（确认用户是要解决宽带问题，及识别故障类型）
def intent_recognize_node(user_input: str):
    # 记录代理开始处理查询
    logger.info("意图识别开始...")
    try:
        tools = [add_human_in_the_loop(human_input)]
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
        react_prompt = get_prompt(Config.PROMPT_INTENT_RECOGNIZE, IntentRecognizeResult)
        #进线后由机器人先询问用户，判断如果历史消息中没有用户消息
        agent = create_agent(
            model=llm_chat,
            tools=tools,
            #checkpointer=checkpointer,
            #response_format={"type": "json_object"},
            prompt=react_prompt
        )

        agent_res = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print(agent_res)
        return agent_res
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
def main():
    try:
        # 打印机器人就绪提示
        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
        # 定义运行时配置，包含线程ID和用户ID
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}
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
            intent_recognize_node(user_input)
    except Exception as e:
        logger.error(f"Graph creation failed: {e}")
        print(f"错误: {e}")

# 检查是否为主模块运行
if __name__ == "__main__":
    # 调用主函数
    main()