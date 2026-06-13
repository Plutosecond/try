"""
CS课程RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "../cs_course_rag_cards.md"
    index_save_path: str = "./vector_index"

    # 模型配置（DashScope embedding + 千问LLM）
    embedding_model: str = "text-embedding-v1"
    llm_model: str = "qwen-plus-2025-07-28"

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


DEFAULT_CONFIG = RAGConfig()
