from langgraph.constants import START,END
from typing import Annotated,TypedDict,Optional
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from config.key import qianwen_key
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool

graph_checkpointer=InMemorySaver()
agent_checkpointer=InMemorySaver()

llm=ChatOpenAI(
    model="qwen-plus",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

graph_config={
    "configurable": {
        "thread_id": "111"
    }
}
agent_config={
    "configurable": {
        "thread_id": "222"
    }
}

def get_weather(city:str):
    """获取某个城市的天气
    Args:
        city:具体城市
    """
    return "城市"+city+"今天天气不错。"

# 使用StructuredTool声明tool
weatherTool=StructuredTool.from_function(func=get_weather,description="获取某个城市天气",name="get_weather")

agent=create_react_agent(
        model=llm,
        tools=[weatherTool],
        prompt="简洁回答问题。",
        checkpointer=agent_checkpointer
    )

class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]
    user_in:str

def call_model(state:State):
    prompt=state["user_in"]
    response=agent.invoke({"messages":HumanMessage(content=prompt)},config=agent_config)
    # print(response["messages"])
    return {"messages":response["messages"]}

builder=StateGraph(State)
builder.add_node("call_model",call_model)
builder.add_edge(START,"call_model")
builder.add_edge("call_model",END)

graph=builder.compile(checkpointer=graph_checkpointer)

print("输入你的问题（输入q或quit退出）：")
while True:
    user_in=input()
    if user_in.lower() in {"q","quit"}:
        break
    # message=HumanMessage(content=user_in)
    # result=graph.invoke({"messages":message},config=config)
    # print(result)
    result=graph.stream({"user_in":user_in},config=graph_config,stream_mode="messages")
    for chunk,metadata in result:
        if metadata['langgraph_node'] != 'tools' and chunk.content:  ##只输出aimessage
            print(chunk.content, end="", flush=True)
        # print(metadata,end="")
    print()
