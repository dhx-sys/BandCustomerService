from contextlib import asynccontextmanager
from typing import TypedDict, Annotated, Sequence, Optional

import uvicorn
from fastapi import FastAPI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from functools import partial
# 修复BaseModel导入，不要从openai导入
from pydantic import BaseModel

from config import Config
from llms import get_llm


class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 请求体模型
class AgentRequest(BaseModel):
    user_id: str
    session_id: str
    query: str


# 恢复中断专用请求体
class AgentResumeRequest(BaseModel):
    user_id: str
    session_id: str
    resume_content: str


def intent_recognize_node(state: MessagesState, config: RunnableConfig):
    # 触发人机中断，阻塞流程等待前端传入用户回复
    interrupt("您好，请问您的宽带遇到什么问题了吗？")
    # 中断恢复后会走到这里，可写后续业务逻辑
    return state


def create_graph(llm_chat) -> StateGraph:
    workflow = StateGraph(MessagesState)
    workflow.add_node("intent_recognize_node", intent_recognize_node)

    workflow.add_edge(START, "intent_recognize_node")
    workflow.add_edge("intent_recognize_node", END)

    # 内存持久化检查点，保存每个session对话状态
    memory_saver = InMemorySaver()
    return workflow.compile(checkpointer=memory_saver)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动初始化Agent图
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    app.agent_graph = create_graph(llm_chat)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/agent/invoke")
async def invoke_agent(request: AgentRequest):
    """首次发起对话，会触发GraphInterrupt返回提示语"""
    session_id = request.session_id
    thread_config = {
        "configurable": {
            "user_id": request.user_id,
            "thread_id": session_id
        }
    }
    initial_state = {
        "messages": [{"role": "user", "content": request.query}]
    }
    try:
        await app.agent_graph.ainvoke(initial_state, config=thread_config)
        return {"code": 1, "msg": "对话流程正常结束"}
    except GraphInterrupt as e:
        # 捕获中断，返回弹窗/提问文本给前端
        return {"code": 2, "msg": "触发人工确认", "interrupt_prompt": e.value}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"code": 3, "msg": f"服务异常: {str(e)}"}


@app.post("/agent/resume")
async def resume_agent(req: AgentResumeRequest):
    """中断后，传入用户回答恢复图执行"""
    thread_config = {
        "configurable": {
            "user_id": req.user_id,
            "thread_id": req.session_id
        }
    }
    try:
        # resume 传入用户回复，继续运行图
        result = await app.agent_graph.ainvoke(None, config=thread_config, resume=req.resume_content)
        return {"code": 1, "data": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"code": 3, "msg": f"恢复执行失败: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)