import os
from typing import TypedDict, Annotated, List, Optional
import operator

from langchain.chat_models import init_chat_model
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver # 导入持久化器
from pydantic import BaseModel, Field

# ======= Model =======
"""模型"""
model = init_chat_model(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# ======= State =======
"""状态"""
class AgentState(TypedDict):
    # 使用 Annotated 和 operator.add，确保新消息是追加而不是覆盖
    messages: Annotated[List[dict], operator.add] 
    need_search: bool
    search_query: Optional[str]
    search_result: Optional[str]


# ======= Pydantic =======
"""约束输出判断文本"""
class DecideOutput(BaseModel):
    need_search: bool = Field(description="是否需要搜索")
    search_query: Optional[str] = Field(default=None, description="搜索关键词")
    direct_answer: Optional[str] = Field(default=None, description="若不搜索，给出的直接回答")


# ======= Tools =======
"""本地文档"""
DOCUMENTS = {
    "langgraph": "LangGraph 是一个用于构建多步、有状态 Agent 的状态机框架。",
    "agent": "Agent 是能够进行决策、调用工具并完成任务的智能体。",
    "memory": "Memory 用于记录用户的长期偏好和背景信息。",
}

"""搜索工具"""
def search_tool(query: str) -> str:
    results = [v for k, v in DOCUMENTS.items() if query.lower() in k.lower() or query.lower() in v.lower()]
    return "\n".join(results) if results else "未找到相关信息"


# ======= Nodes =======
"""判断边逻辑：决定去搜索还是直接回答"""
def decide_node(state: AgentState):
    # 只需要最后一条用户消息来做决策
    decision = model.with_structured_output(DecideOutput).invoke(state["messages"])
    
    return {
        "need_search": decision.need_search,
        "search_query": decision.search_query,
        # 如果不需要搜索，我们暂时把回复存入 result 传递给下游，或者直接这里返回消息
        "search_result": decision.direct_answer if not decision.need_search else None
    }

"""搜索边逻辑"""
def search_node(state: AgentState):
    result = search_tool(state["search_query"])
    return {"search_result": result}

"""最终生成回复节点"""
def respond_node(state: AgentState):
    # 如果是搜索出来的结果，让 AI 结合背景知识回答
    if state.get("need_search"):
        prompt = f"根据以下参考信息回答用户：\n{state['search_result']}"
        # 构造一个临时系统消息发送给模型
        combined_msgs = state["messages"] + [{"role": "system", "content": prompt}]
        response = model.invoke(combined_msgs)
        return {"messages": [{"role": "assistant", "content": response.content}]}
    else:
        # 如果是直接回答的结果（从 decide 传过来的）
        return {"messages": [{"role": "assistant", "content": state["search_result"]}]}


# ======= 构建图 =======
"""创建状态图"""
workflow = StateGraph(AgentState)

"""加入节点"""
workflow.add_node("decide", decide_node)
workflow.add_node("search", search_node)
workflow.add_node("respond", respond_node)

"""加入边"""
workflow.add_edge(START, "decide")

def route_after_decide(state: AgentState):
    if state["need_search"]:
        return "search"
    return "respond"

workflow.add_conditional_edges(
    "decide",
    route_after_decide,
    {
        "search": "search",
        "respond": "respond"
    }
)

workflow.add_edge("search", "respond")
workflow.add_edge("respond", END)

"""使用 MemorySaver 实现多轮对话持久化"""
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# ======= 使用 =======
if __name__ == "__main__":
    # thread_id 是多轮对话的关键
    config = {"configurable": {"thread_id": "user_123"}}
    
    print("Agent 已启动（输入 exit 退出）")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            break

        # 只需要传入新消息，之前的消息会自动从 checkpointer 加载
        input_state = {"messages": [{"role": "user", "content": user_input}]}
        
        # 传入 config
        final_state = app.invoke(input_state, config=config)
        
        print("Agent:", final_state["messages"][-1]["content"])