import datetime

from langchain_openai import ChatOpenAI
from config.key import qianwen_key
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.tools import tool
model = ChatOpenAI(
    model="qwen-plus-2025-07-28",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@tool
def get_date():
    """获取今天日期"""
    return datetime.datetime.today().strftime("%Y-%m-%d")

model=model.bind_tools([get_date])
all_tools={"get_date":get_date}

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
    stream=model.stream(messages)

    str=""
    # stream=model.stream([SystemMessage(content=system_in),HumanMessage(content=human_in)])
    for chunk in stream:
        str+=chunk.content
        print(chunk.content,end="")

    history.add_user_message(user_in)
    history.add_ai_message(str)

    print("\n")