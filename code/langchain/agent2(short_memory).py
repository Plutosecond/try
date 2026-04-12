from langgraph.prebuilt import create_react_agent
from config.key import qianwen_key
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain.tools import tool,StructuredTool

checkpointer=InMemorySaver()

model=ChatOpenAI(
    model="qwen-plus",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def get_weather(city:str):
    """获取某个城市的天气
    Args:
        city:具体城市
    """
    return "城市"+city+"今天天气不错。"

# 使用StructuredTool声明tool
weatherTool=StructuredTool.from_function(func=get_weather,description="获取某个城市天气",name="get_weather")



agent=create_react_agent(
    model=model,
    tools=[weatherTool],
    prompt="简要回答问题。",
    checkpointer=checkpointer
)

config={
    "configurable":{
        "thread_id":"1"
    }
}
print("你可以问一些问题：")
while True:
    user_in=input()
    if user_in.lower() in ("q","quit"):
        break
    messages=HumanMessage(content=user_in)

    result = agent.stream({"messages": messages}, stream_mode="messages",config=config)
    for msg_chunk, metadata in result:
        # print(chunk, end="\n", flush=True)
        if metadata['langgraph_node'] == 'agent' and msg_chunk.content:  ##只输出aimessage
            print(msg_chunk.content, end="", flush=True)
    print()