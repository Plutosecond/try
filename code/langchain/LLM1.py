from langchain_openai import ChatOpenAI
from config.key import openai_key
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_key,
    stream=True
)

system_in=f"""
    你的性格十分活泼，用欢快的口吻回答问题。
"""
human_in=f"""
    简短介绍一下你自己。
"""
stream=model.stream([SystemMessage(content=system_in),HumanMessage(content=human_in)])
for chunk in stream:
    print(chunk.content,end="")