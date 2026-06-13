"""
索引构建模块 —— 使用 DashScope 嵌入模型 + FAISS 向量存储
"""

import logging
from typing import List, Optional
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """索引构建模块 - 负责文本向量化与FAISS索引的构建/保存/加载"""

    def __init__(self, model_name: str = "text-embedding-v1", index_save_path: str = "./vector_index"):
        """
        初始化索引构建模块

        Args:
            model_name: DashScope 嵌入模型名称，默认 text-embedding-v1
            index_save_path: 向量索引持久化保存目录
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings: Optional[DashScopeEmbeddings] = None
        self.vectorstore: Optional[FAISS] = None
        self._setup_embeddings()

    def _setup_embeddings(self):
        """初始化 DashScope 嵌入模型（阿里云通义千问 Embedding 服务）"""
        logger.info("正在初始化 DashScope 嵌入模型: %s", self.model_name)

        # DashScopeEmbeddings 自动读取环境变量 DASHSCOPE_API_KEY
        # 底层调用阿里云 text-embedding-v1 接口，输出 1536 维向量
        self.embeddings = DashScopeEmbeddings(
            model=self.model_name,
        )

        logger.info("DashScope 嵌入模型初始化完成")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        将文档块列表构建为 FAISS 向量索引

        工作流程：
        1. 遍历每个文档块，调用 DashScope API 获取对应 Embedding 向量
        2. 将所有向量写入 FAISS 内存索引（默认使用 L2 归一化 + 内积搜索）
        3. 索引对象保存在 self.vectorstore 中，供后续检索使用

        Args:
            chunks: 文档块列表（由 DataPreparationModule.chunk_documents() 产出）

        Returns:
            FAISS 向量存储对象
        """
        logger.info("正在构建 FAISS 向量索引...")

        if not chunks:
            raise ValueError("文档块列表不能为空，请先调用数据准备模块的分块方法")

        # FAISS.from_documents 内部会：
        #   - 调用 self.embeddings.embed_documents() 批量生成向量
        #   - 构建 IndexFlatIP（内积索引）或 IndexFlatL2（L2索引）
        #   - 将文档内容与元数据一并存入 docstore
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        logger.info("FAISS 向量索引构建完成，共 %d 个向量", len(chunks))
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        """
        向已有索引增量添加新文档（增量更新，无需重建整个索引）

        Args:
            new_chunks: 新增的文档块列表
        """
        if not self.vectorstore:
            raise ValueError("请先调用 build_vector_index() 构建初始索引")

        logger.info("正在增量添加 %d 个文档块...", len(new_chunks))
        self.vectorstore.add_documents(new_chunks)
        logger.info("增量添加完成")

    def save_index(self):
        """
        将当前 FAISS 索引持久化到磁盘

        保存目录由 self.index_save_path 指定，包含两个文件：
          - index.faiss ：FAISS 二进制向量索引
          - index.pkl   ：文档内容与元数据的 pickle 文件
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引后再保存")

        save_dir = Path(self.index_save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(str(save_dir))
        logger.info("向量索引已保存到: %s", save_dir)

    def load_index(self) -> Optional[FAISS]:
        """
        从磁盘加载已持久化的 FAISS 索引

        如果索引目录不存在或加载失败，返回 None，
        调用方此时应走"构建新索引"的完整流程。

        Returns:
            加载成功的 FAISS 对象，或 None（索引不存在/加载失败）
        """
        if not self.embeddings:
            self._setup_embeddings()

        save_dir = Path(self.index_save_path)
        if not save_dir.exists():
            logger.info("索引路径 %s 不存在，需要构建新索引", save_dir)
            return None

        try:
            # allow_dangerous_deserialization=True 是 langchain FAISS 加载的必要参数
            # 仅在信任索引来源（自己构建的）时使用
            self.vectorstore = FAISS.load_local(
                str(save_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("向量索引已从 %s 成功加载", save_dir)
            return self.vectorstore
        except Exception as e:
            logger.warning("加载向量索引失败: %s，将重新构建", e)
            return None

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        最基础的相似度检索（直接使用向量余弦相似度）

        Args:
            query: 查询文本
            k: 返回的 Top-K 文档数量

        Returns:
            按相似度降序排列的文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search(query, k=k)
