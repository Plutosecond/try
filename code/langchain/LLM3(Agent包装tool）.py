import datetime
from langchain_openai import ChatOpenAI
from config.key import qianwen_key
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.tools import tool,StructuredTool
from langchain.agents import  initialize_agent,AgentType
# agent历史记忆
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# 还没完成agent封装的多轮记忆


model = ChatOpenAI(
    model="qwen-plus-2025-07-28",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

memory=ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
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

# all_tools={"get_date":get_date,"get_weather":weatherTool}
# model=model.bind_tools([get_date,weatherTool])


agent_executor = initialize_agent(
    tools=[get_date, weatherTool],
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
    agent_kwargs={
        "system_message": "用简洁的口吻回答问题。"
    }
)

# history=InMemoryChatMessageHistory()
print("这是一个大语言模型，你可以输入你的问题：")

while True:
    user_in=input()

    if user_in.lower() in ['exit', 'quit', '退出']:
        print("再见！")
        break
    if not user_in:
        continue

    system_in=f"""
        用简洁的口吻回答问题。
    """
    # history.add_ai_message(system_in)
    human_in=f"""
        {user_in}
    """
    # messages = [SystemMessage(content=system_in)]
    # messages.extend(history.messages)
    # messages.append(HumanMessage(content=human_in))


    ai_msg=agent_executor.invoke({"input": human_in})
    response=ai_msg["output"]

    print(f"回答：{response}")
    # print(f"当前记忆: {memory.chat_memory.messages}\n")
    # history.add_user_message(user_in)
    # history.add_ai_message(response)

    print("\n")