from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

from config.key import qianwen_key
from langchain_openai import ChatOpenAI


os.environ["DASHSCOPE_API_KEY"] = qianwen_key
persist_dir = "chroma_db"
pdf_path="data.pdf"



embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
)
##################################################################################
##文本分割
text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=80)

#################################################################################
##向量生成
if not os.path.exists(persist_dir):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(docs[0].page_content[:200])
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
else:
    # print(docs[0].page_content[:200])
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
##############################################################################
##检索生成
llm = ChatOpenAI(
    model="qwen-plus-2025-07-28",
    api_key=qianwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

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
    create_stuff_documents_chain(llm, prompt),
)

while True:
        question = input("\n问题（q 退出）：").strip()
        if question.lower() in ("q", "quit"):
            break
        for chunk in qa_chain.stream({"input": question}):
            # print(chunk,end="",flush=True)
            if "answer" in chunk:
                print(chunk["answer"],end="",flush=True)
        print()
