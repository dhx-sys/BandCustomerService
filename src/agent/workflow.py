from langgraph.graph import START, END, StateGraph

from src.agent.nodes import intent_recognize_node, get_user_info, get_band_info_by_idcard, check_band_info, \
    router_after_get_user_info_node, router_after_get_band_info_by_idcard, get_fault_code

from functools import partial

from src.agent.state import MessagesState


def create_graph(llm_chat, checkpointer)-> StateGraph:
    #创建状态图实例，使用MessagesState作为状态类型
    workflow = StateGraph(MessagesState)
    workflow.add_node("intent_recognize_node", partial(intent_recognize_node, llm_chat=llm_chat))
    workflow.add_node("get_user_info", partial(get_user_info, llm_chat=llm_chat))
    workflow.add_node("get_band_info_by_idcard", partial(get_band_info_by_idcard, llm_chat=llm_chat))
    workflow.add_node("check_band_info", partial(check_band_info, llm_chat=llm_chat))
    workflow.add_node("get_fault_code", partial(get_fault_code, llm_chat=llm_chat))
    # 添加从起始到代理的边
    workflow.add_edge(START, "intent_recognize_node")
    workflow.add_edge("intent_recognize_node","get_user_info")
    workflow.add_conditional_edges("get_user_info", lambda state: router_after_get_user_info_node(state))
    workflow.add_conditional_edges("get_band_info_by_idcard", lambda state: router_after_get_band_info_by_idcard(state))
    workflow.add_edge("check_band_info", "get_fault_code")
    workflow.add_edge("get_fault_code", END)

    # 编译状态图，绑定检查点和存储
    return workflow.compile(checkpointer=checkpointer)




