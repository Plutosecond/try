from Director import graph

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
