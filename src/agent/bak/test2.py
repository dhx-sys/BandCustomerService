from contextlib import asynccontextmanager
from typing import TypedDict, Annotated, Sequence

import uvicorn
from fastapi import FastAPI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from functools import partial

from openai import BaseModel

from config import Config
from llms import get_llm


class MessagesState(TypedDict):
    # 定义messages字段，类型为消息序列使用add_messages处理追加，
    messages: Annotated[Sequence[BaseMessage], add_messages]


class AgentRequest(BaseModel):
    user_id: str
    session_id: str
    query: str


def intent_recognize_node(state: MessagesState, config: RunnableConfig):
    # 记录代理开始处理查询
    # 触发中断
    assistant = "您好，请问您的宽带遇到什么问题了吗？"
    interrupt_response = interrupt("您好，请问您的宽带遇到什么问题了吗？")
    return interrupt_response


def create_graph(llm_chat) -> StateGraph:
    #创建状态图实例，使用MessagesState作为状态类型
    workflow = StateGraph(MessagesState)
    workflow.add_node("intent_recognize_node", partial(intent_recognize_node))

    # 添加从起始到代理的边
    workflow.add_edge(START, "intent_recognize_node")
    workflow.add_edge("intent_recognize_node", END)

    # 编译状态图，绑定检查点和存储
    return workflow.compile(checkpointer=InMemorySaver())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时创建图实例
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    app.agent = create_graph(llm_chat)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/agent/invoke")
async def invoke_agent(request: AgentRequest):
    session_id = request.session_id
    thread_config = {"configurable": {"user_id": "user123", "thread_id": session_id}}
    # 构造初始状态
    initial_state = {
        "messages": [{"role": "user", "content": request.query}]
    }
    try:
        # 调用图
        # ainvoke 会在遇到 interrupt 时抛出 GraphInterrupt 异常（在大多数 0.2+ 版本中）
        result = await app.agent.invoke(initial_state, config=thread_config, version="v2")
        # 如果没有异常，说明执行完成
        return "1"
    except GraphInterrupt as e:
        # 捕获中断异常
        # e.value 是 interrupt(value=...) 中传入的内容
        return "2"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "3"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
