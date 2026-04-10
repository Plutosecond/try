import datetime
from langchain_openai import ChatOpenAI
from config.key import qianwen_key
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.tools import tool,StructuredTool
model = ChatOpenAI(
    model="qwen-plus-2025-07-28",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 显式声明工具
@tool
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

all_tools={"get_date":get_date,"get_weather":weatherTool}
model=model.bind_tools([get_date,weatherTool])


history=InMemoryChatMessageHistory()
print("这是一个大语言模型，你可以输入你的问题：")

while True:
    user_in=input()

    if user_in.lower() in ['exit', 'quit', '退出']:
        print("再见！")
        break

    system_in=f"""
        用简洁的口吻回答问题。
    """
    # history.add_ai_message(system_in)
    human_in=f"""
        {user_in}
    """
    messages = [SystemMessage(content=system_in)]
    messages.extend(history.messages)
    messages.append(HumanMessage(content=human_in))
    # history.add_user_message(human_in)
    # print(history.messages)
    check_tools=model.invoke(messages)

    # print(ai_msg.content)

    if check_tools.tool_calls:
        messages.append(check_tools)
        for tool_call in check_tools.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name in all_tools:
                result = all_tools[tool_name].invoke(tool_args)
                messages.append(ToolMessage(content=result, tool_call_id=tool_id))

    stream=model.stream(messages)
    response=""
    for chunk in stream:
        if chunk.content:
            response+=chunk.content
            print(chunk.content, end="", flush=True)

    history.add_user_message(user_in)
    history.add_ai_message(response)

    print("\n")