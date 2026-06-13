"""
数据准备模块 —— CS课程知识卡片
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import uuid

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """数据准备模块 - 加载CS课程Markdown知识卡片，解析元数据并按标题分块"""

    DOMAIN_LABELS = ["AI", "系统", "软件工程", "理论计算机", "数据科学", "网络安全", "计算机视觉与图形学", "网络"]
    DIFFICULTY_LABELS = ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"]
    IMPORTANCE_LABELS = ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"]

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []   # 父文档（完整课程）
        self.chunks: List[Document] = []       # 子文档（按H2标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def load_documents(self) -> List[Document]:
        """加载并解析整个课程Markdown文件，每门课程作为一个父文档"""
        logger.info(f"正在加载课程数据: {self.data_path}")

        raw_text = Path(self.data_path).read_text(encoding="utf-8")
        # 按 # 一级标题拆分课程（跳过文件开头的空白）
        courses = self._split_by_h1(raw_text)

        documents: List[Document] = []
        for title, body in courses:
            if "课程关系总览" in title:          # 末尾的关系图不是独立课程
                continue
            parent_id = str(uuid.uuid4())
            doc = Document(
                page_content=body,
                metadata={
                    "source": self.data_path,
                    "course_name": title.strip(),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                },
            )
            self._enhance_metadata(doc)
            documents.append(doc)

        self.documents = documents
        logger.info("成功加载 %d 门课程", len(documents))
        return documents

    def chunk_documents(self) -> List[Document]:
        """Markdown结构感知分块（按H2标题切分各课程）"""
        logger.info("正在进行Markdown结构感知分块...")
        if not self.documents:
            raise ValueError("请先调用 load_documents()")

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "h2_header"),
                ("###", "h3_header"),
            ],
            strip_headers=False,
        )

        all_chunks: List[Document] = []
        for doc in self.documents:
            # 重建完整markdown以便分割器正确解析
            full_md = f"# {doc.metadata['course_name']}\n{doc.page_content}"
            try:
                sub_chunks = splitter.split_text(full_md)
            except Exception as e:
                logger.warning("课程 %s 分割失败: %s", doc.metadata.get("course_name"), e)
                sub_chunks = [doc]

            for i, chunk in enumerate(sub_chunks):
                child_id = str(uuid.uuid4())
                parent_id = doc.metadata["parent_id"]
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": i,
                })
                self.parent_child_map[child_id] = parent_id
                all_chunks.append(chunk)

        self.chunks = all_chunks
        logger.info("Markdown分块完成，共生成 %d 个chunk", len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------
    # 元数据相关
    # ------------------------------------------------------------------

    def _enhance_metadata(self, doc: Document):
        """从课程正文中提取领域、难度、标签、求职重要程度等"""
        content = doc.page_content

        doc.metadata["domain"] = self._extract_field(content, r"所属领域[：:]\s*(.+)")
        doc.metadata["difficulty"] = self._extract_field(content, r"课程难度[：:]\s*(.+)")
        doc.metadata["job_importance"] = self._extract_field(content, r"求职重要程度[：:]\s*(.+)")
        doc.metadata["tags"] = self._extract_tags(content)

    def _extract_field(self, text: str, pattern: str) -> str:
        m = re.search(pattern, text)
        return m.group(1).strip() if m else "未知"

    def _extract_tags(self, text: str) -> List[str]:
        m = re.search(r"课程标签[：:]\s*(.+)", text)
        if not m:
            return []
        return [t.strip().lstrip("#") for t in m.group(1).split() if t.strip().startswith("#")]

    # ------------------------------------------------------------------
    # 查询辅助
    # ------------------------------------------------------------------

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """根据检索到的子块获取去重后的父文档（按匹配次数排序）"""
        relevance: Dict[str, int] = {}
        parent_map: Dict[str, Document] = {}

        for chunk in child_chunks:
            pid = chunk.metadata.get("parent_id")
            if not pid:
                continue
            relevance[pid] = relevance.get(pid, 0) + 1
            if pid not in parent_map:
                for doc in self.documents:
                    if doc.metadata.get("parent_id") == pid:
                        parent_map[pid] = doc
                        break

        sorted_pids = sorted(relevance, key=lambda p: relevance[p], reverse=True)
        return [parent_map[pid] for pid in sorted_pids if pid in parent_map]

    def filter_by_domain(self, domain: str) -> List[Document]:
        return [d for d in self.documents if d.metadata.get("domain") == domain]

    def filter_by_difficulty(self, difficulty: str) -> List[Document]:
        return [d for d in self.documents if d.metadata.get("difficulty") == difficulty]

    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}

        domains: Dict[str, int] = {}
        difficulties: Dict[str, int] = {}
        for doc in self.documents:
            d = doc.metadata.get("domain", "未知")
            domains[d] = domains.get(d, 0) + 1
            diff = doc.metadata.get("difficulty", "未知")
            difficulties[diff] = difficulties.get(diff, 0) + 1

        return {
            "total_courses": len(self.documents),
            "total_chunks": len(self.chunks),
            "domains": domains,
            "difficulties": difficulties,
        }

    @classmethod
    def get_supported_domains(cls) -> List[str]:
        return cls.DOMAIN_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        return cls.DIFFICULTY_LABELS

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_h1(text: str) -> List[tuple]:
        """将markdown按一级标题拆分为 (标题, 正文) 列表"""
        # 使用正则匹配行首的 # 标题（但不匹配 ## 等）
        parts = re.split(r"\n(?=# (?!#))", text)
        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines = part.split("\n", 1)
            title = lines[0].lstrip("#").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            result.append((title, body))
        return result
