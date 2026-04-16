from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.constants import START,END
from typing import Annotated,TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage,HumanMessage,SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from config.key import qianwen_key
from langchain.tools import tool,StructuredTool

nodes=["GetWeather", "PaperReader", "Joker", "Other"]

persist_dir = "chroma_db"
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=qianwen_key
)

class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]
    user_in:str
    type:str

builder=StateGraph(State)

model=ChatOpenAI(
    model="qwen-plus",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

graph_checkpointer=InMemorySaver()
agent_checkpointer=InMemorySaver()

def get_weather(city:str):
    """获取某个城市的天气
    Args:
        city:具体城市
    """
    return "城市"+city+"今天天气不错。"

weatherTool=StructuredTool.from_function(func=get_weather,description="获取某个城市天气",name="get_weather")

agent=create_react_agent(
    model=model,
    tools=[weatherTool],
    checkpointer=agent_checkpointer
)

def SuperVisor(state:State):
    print(">>>Supervisor_node")
    writer=get_stream_writer()
    writer({"node":">>>SuperVisor_node"})

    user_in = state["user_in"]
    system_prompt = "你将根据用户输入判断用户意图：" \
                    "获取某地天气： 返回 GetWeather" \
                    "获取论文相关内容：返回 PaperReader" \
                    "获取笑话：返回 Joker" \
                    "其他： 返回 Other" \
                    "注意只要返回所提供的英文字母，不要有任何其他东西！"
    human_prompt = f"用户输入：{user_in}" \
                   f"你的回答："
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    result = model.invoke(messages)
    # result="Joker"
    writer({"result":f">>>{result.content}"})

    if result.content not in nodes:
        raise ValueError

    return {"type":result.content}

def GetWeather(state:State):
    print(">>>GetWeather_node")
    writer = get_stream_writer()
    writer({"node": ">>>GetWeather_node"})
    user_in=state["user_in"]
    response = agent.invoke({"messages": HumanMessage(content=user_in)})##这里可以加config实现这个节点agent的多轮记忆，所以以后要实现langgraph的多轮记忆可以全用agent（checkpoint+config）方便
    writer({"response": f">>>{response}"})
    # print(response["messages"])
    return {}

def PaperReader(state:State):
    print(">>>PaperReader_node")
    writer = get_stream_writer()
    writer({"node": ">>>PaperReader_node"})

    question=state["user_in"]

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    ##############################################################################
    ##检索生成

    system_prompt = (
        "你是一个文档问答助手。根据以下检索到的上下文片段回答问题，"
        "如果上下文中没有相关信息，请直接说不知道，不要编造答案。\n\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "上下文：\n{context}\n\n问题：{input}"),
    ])

    qa_chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(model, prompt),
    )
    result=qa_chain.invoke({"input": question})
    writer({"result": f">>>{result}"})
    return {}

def Joker(state:State):
    print(">>>Joker_node")
    writer = get_stream_writer()
    # writer({"node": ">>>Joker_node"})
    sys_in=SystemMessage(content="你是郭德纲复制体，熟悉所有他的思想、语言、幽默感，以他的风格给出100字以内的笑话。")
    user_in=state["user_in"]
    response=model.invoke([sys_in,HumanMessage(content=user_in)])
    # writer({"response":f">>>{response.content}"})
    return {"messages": response.content}

def Other(state:State):
    print(">>>Other_node")
    writer = get_stream_writer()
    writer({"node": ">>>Other_node"})
    # user_in=state["user_in"]
    # response=model.invoke([HumanMessage(content=user_in)])
    # writer({"response":f"{response.content}"})
    message=AIMessage(content="抱歉我无法回答你的问题。")
    return {"messages":message}
# LangGraph 的 stream_mode="messages" 会流式输出所有添加到 messages 中的消息，包括：
# AIMessage\HumanMessage\ToolMessage

def rout_func(state:State):
    type=state["type"]
    return str(type)

builder.add_node("SuperVisor",SuperVisor)
builder.add_node("GetWeather",GetWeather)
builder.add_node("PaperReader",PaperReader)
builder.add_node("Joker",Joker)
builder.add_node("Other",Other)

builder.add_edge(START,"SuperVisor")
builder.add_conditional_edges("SuperVisor",rout_func,["GetWeather","PaperReader","Joker","Other"])
builder.add_edge("GetWeather",END)
builder.add_edge("PaperReader",END)
builder.add_edge("Joker",END)
builder.add_edge("Other",END)

graph=builder.compile(checkpointer=graph_checkpointer)



if __name__=="__main__":

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    # result=graph.stream({"user_in":"文章在电网方面做了什么研究"},stream_mode="messages",config=config)
    # for chunk,metadata in result:
    #     if metadata['langgraph_node'] != 'tools' and metadata['langgraph_node'] != 'SuperVisor' and chunk.content:
    #         print(chunk.content, end="", flush=True)
    print("多轮对话开始（输入 'quit' 退出）：")

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ['quit', 'q']:
            break

        print("助手: ", end="", flush=True)

        result = graph.stream({"user_in": user_input}, stream_mode="messages", config=config)

        for chunk, metadata in result:
            if (metadata['langgraph_node'] != 'tools' and
                    metadata['langgraph_node'] != 'SuperVisor' and
                    chunk.content):
                print(chunk.content, end="", flush=True)

        print()