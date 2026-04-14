from typing import TypedDict, Annotated
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing import Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage
from operator import add
from langgraph.checkpoint.memory import InMemorySaver

checkpointer=InMemorySaver()

class FullState(TypedDict):
    u1: int
    s1: Optional[str]  ##can be null,下面就要用state.get()
    s2: Optional[str]
    u2: int
    messages: Annotated[list[AnyMessage], add_messages]  ##后续出现的messages直接add到列表之后而不是覆盖
    list: Annotated[list[int], add]  ##后续出现的int直接add到列表之后而不是覆盖


class Input_State(TypedDict):  ##状态定义一定要是字典型
    u1: int
    u2: int


class Output_State(TypedDict):  ##可以有不同状态分配给不同node
    s1: str
    s2: str
    messages: Annotated[list[AnyMessage], add_messages]  ##后续出现的messages直接add到列表之后而不是覆盖
    list: Annotated[list[int], add]  ##后续出现的int直接add到列表之后而不是覆盖


##每个节点就是个函数，后续会带上llm
def node1(state: FullState) -> FullState:  ##第一个是输入状态，第二个是返回状态（返回的是返回状态中的字典）
    message = AIMessage(content=f"{state['u1']}okokokokok")
    lis = state["u1"] * 2
    return {"u1": state["u1"], "u2": state["u2"], "s1": state.get("s1"), "s2": state.get("s2"), "messages": [message],
            "list": [lis]}


def node2(state: FullState) -> dict:  ##dict表示部分更新或者直接不写返回类型
    message = AIMessage(content=f"{state['u2']}okokokokok")
    lis = state["u2"] * 2
    return {"s1": f"基于 u1={state['u1']} 生成", "s2": f"基于 u2={state['u2']} 生成", "messages": [message],
            "list": [lis]}

config={
    "configurable":{
        "thread_id":"111"
    }
}


builder = StateGraph(FullState, input=Input_State, output=Output_State)

builder.add_node("node1", node1)
builder.add_node("node2", node2)

builder.add_edge(START, "node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

graph = builder.compile(checkpointer=checkpointer)##short_term_memory

initial_state = {"u1": 10, "u2": 20}

result = graph.invoke(initial_state, config=config)

print(result)##最后返回的是output状态如果没有指明就是全部状态
