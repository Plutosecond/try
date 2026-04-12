from langchain_openai import ChatOpenAI
from config.key import qianwen_key
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool,StructuredTool
from langgraph.store.memory import InMemoryStore

long_term_store=InMemoryStore()

model=ChatOpenAI(
    model="qwen-plus",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@tool
def remember(key: str, value: str, user_id: str) -> str:
    """记住信息"""
    long_term_store.put(("user", user_id), key, {"value": value})
    return f"记住了：{key}={value}"

@tool
def recall(key: str, user_id: str) -> str:
    """回忆信息"""
    mem = long_term_store.get(("user", user_id), key)
    if mem:
        return f"{key}是：{mem.value['value']}"
    return f"没有找到{key}的信息"

agent=create_react_agent(
    model=model,
    tools=[remember,recall],
    store=long_term_store,
    prompt="""你负责记忆和管理用户信息。
    示例：用户说"我叫小明" → remember(key="name", value="小明", user_id=user_id)
    用户问"我叫什么" → recall(key="name", user_id=user_id)"""
)

user_id = input("用户ID: ").strip()
config = {"configurable": {"thread_id": "1", "user_id": user_id}}

while True:
    user_input = input("\n ")
    if user_input.lower() == 'q':
        break

    messages = [
        {"role": "system", "content": f"user_id={user_id}"},
        HumanMessage(content=user_input)
    ]

    # 使用 values 模式
    result = agent.stream(
        {"messages": messages},
        stream_mode="values",
        config=config
    )

    for chunk in result:
        if "messages" in chunk:
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, 'content') and last_msg.type == "ai":
                print(f"助手: {last_msg.content}")