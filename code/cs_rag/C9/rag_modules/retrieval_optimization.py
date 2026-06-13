"""
检索优化模块 —— 混合检索（向量 + BM25） + RRF 重排序 + 元数据过滤
"""

import logging
import hashlib
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """检索优化模块 - 融合向量语义检索与 BM25 关键词检索，提升召回质量"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块

        Args:
            vectorstore: FAISS 向量存储（由 IndexConstructionModule 构建）
            chunks: 全量文档块列表（用于构建 BM25 索引）
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self._setup_retrievers()

    def _setup_retrievers(self):
        """构建双路检索器：FAISS 向量检索器 + BM25 关键词检索器"""
        logger.info("正在初始化双路检索器...")

        # 向量检索器 —— 基于语义相似度，擅长捕获同义/近义表达
        # 例如："操作系统" 能匹配到 "OS原理"、"进程管理" 等内容
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        # BM25 检索器 —— 基于词频-逆文档频率，擅长精确关键词匹配
        # 例如：精确匹配 "B+树"、"动态规划" 等专业术语
        # 底层使用 scikit-learn 的 TfidfVectorizer 实现
        self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)

        logger.info("双路检索器初始化完成（FAISS向量检索 + BM25关键词检索）")

    # ------------------------------------------------------------------
    # 公开检索接口
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 —— 同时执行向量检索与BM25检索，使用RRF算法融合排序

        RRF (Reciprocal Rank Fusion) 原理：
          - 两份排序列表都出现的文档获得更高总分
          - 在某一路检索中排名靠前的文档获得更高权重
          - 即使某文档只被一路检索命中，仍有机会进入最终结果
          - 参数 k=60 用于平滑极端排名差异

        Args:
            query: 用户查询文本
            top_k: 最终返回的文档数量

        Returns:
            经过 RRF 重排序后的 Top-K 文档列表
        """
        # 第一步：并行执行两路检索
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 第二步：RRF 融合排序
        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked[:top_k]

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        """
        带元数据过滤的检索 —— 先混合检索获取候选集，再按元数据条件过滤

        典型用法：
          - 按领域过滤：filters={"domain": "AI"}
          - 按难度过滤：filters={"difficulty": "★★★★☆"}
          - 组合过滤：filters={"domain": "系统", "difficulty": "★★★☆☆"}

        Args:
            query: 用户查询文本
            filters: 元数据过滤条件字典，如 {"domain": "系统", "difficulty": "★★★☆☆"}
            top_k: 最终返回的文档数量

        Returns:
            满足所有过滤条件的 Top-K 文档列表
        """
        # 先取 3 倍候选集，避免过滤后数量不足
        candidates = self.hybrid_search(query, top_k * 3)

        filtered: List[Document] = []
        for doc in candidates:
            match = True
            for key, expected_value in filters.items():
                actual_value = doc.metadata.get(key)
                if actual_value is None or actual_value != expected_value:
                    match = False
                    break
            if match:
                filtered.append(doc)
                if len(filtered) >= top_k:
                    break

        logger.info(
            "元数据过滤检索: query='%s' filters=%s → %d/%d 候选通过过滤",
            query, filters, len(filtered), len(candidates),
        )
        return filtered

    # ------------------------------------------------------------------
    # RRF 排序算法
    # ------------------------------------------------------------------

    def _rrf_rerank(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """
        RRF (Reciprocal Rank Fusion) 重排序算法

        对来自向量检索和 BM25 检索的两份文档列表进行融合排序。
        每篇文档的最终得分 = 在向量检索中的 RRF 分 + 在 BM25 检索中的 RRF 分

        RRF 公式：score(d) = Σ 1 / (k + rank_i(d))
          其中 rank_i(d) 是文档 d 在第 i 路检索中的排名（从 0 开始）

        使用文档内容的 MD5 哈希作为去重标识，确保同一文档在两路中只计一次分。

        Args:
            vector_docs: 向量检索结果列表（已按相似度排序）
            bm25_docs: BM25检索结果列表（已按相关性排序）
            k: RRF 平滑参数，默认 60（学术论文常用值）

        Returns:
            RRF 融合重排后的文档列表（降序）
        """
        doc_scores: Dict[str, float] = {}    # 文档ID → 累计RRF分数
        doc_objects: Dict[str, Document] = {} # 文档ID → Document对象

        # 计算向量检索路的 RRF 分数
        for rank, doc in enumerate(vector_docs):
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)   # rank从0开始，+1保证分母≥1
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

        # 计算 BM25 检索路的 RRF 分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

        # 按 RRF 总分降序排列
        sorted_pairs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        reranked_docs: List[Document] = []
        for doc_id, final_score in sorted_pairs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata["rrf_score"] = round(final_score, 6)
                reranked_docs.append(doc)

        logger.info(
            "RRF重排完成: 向量检索%d篇 + BM25检索%d篇 → 去重合并%d篇",
            len(vector_docs), len(bm25_docs), len(reranked_docs),
        )
        return reranked_docs
