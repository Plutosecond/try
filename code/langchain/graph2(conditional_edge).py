from typing import TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    num: int
    answer: str


def node1(state: State) -> dict:
    return {"answer": "这个数大于5"}


def node2(state: State) -> dict:
    return {"answer": "这个数等于5"}


def node3(state: State) -> dict:
    return {"answer": "这个数小于5"}

###返回节点列表就可以并行
def routing_func(state: State):
    x = state["num"]
    if x > 5:
        return "node1"
    elif x == 5:
        return "node2"
    else:
        return "node3"

builder = StateGraph(State)

builder.add_node(node1)
builder.add_node(node2)
builder.add_node(node3)

builder.add_conditional_edges(START, routing_func, ["node1", "node2", "node3"])
builder.add_edge("node1", END)
builder.add_edge("node2", END)
builder.add_edge("node3", END)

graph=builder.compile()

def test_number(num: int):
    result = graph.invoke({"num": num, "answer": ""})
    print(f"输入: {num} -> {result['answer']}")

test_number(10)
test_number(5)
test_number(3)