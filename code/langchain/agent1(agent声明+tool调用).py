from langgraph.prebuilt import create_react_agent
from config.key import qianwen_key
from langchain_openai import ChatOpenAI
from langchain_core.messages.human import HumanMessage
import datetime
from langchain.tools import tool,StructuredTool

model = ChatOpenAI(
    model="qwen-plus",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@tool("get_current_day",return_direct=False)##直接返回工具值会导致下面流式输出忽略toolmessages
def get_date():
    """获取今天日期"""
    return datetime.datetime.today().strftime("%Y-%m-%d")


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
    tools=[get_date,weatherTool],##在agent中，可以直接传入函数而不用声明工具
    prompt="用简洁的话回复问题",
)

user_in=input("你可以提出一些问题：")

messages=[HumanMessage(content=user_in)]
result=agent.stream({"messages":messages},stream_mode="messages")

for msg_chunk,metadata in result:
    # print(chunk, end="\n", flush=True)
    if metadata['langgraph_node'] == 'agent' and msg_chunk.content:##只输出aimessage
        print(msg_chunk.content, end="", flush=True)
# print(result)
# print(result["messages"][-1].content)