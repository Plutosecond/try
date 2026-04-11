from openai import OpenAI
from config.key import qianwen_key

client = OpenAI(
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

text=input()

query_result = client.embeddings.create(
    model="text-embedding-v1",
    input=text
)

embedding_vector = query_result.data[0].embedding
print(embedding_vector)
