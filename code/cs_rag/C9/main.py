"""
CS课程RAG系统 —— 主程序入口

功能说明：
  基于计算机课程知识卡片（cs_course_rag_cards.md）构建的 RAG 问答系统，
  支持课程查询、领域筛选、学习路线咨询等功能。

技术栈：
  - Embedding：阿里云 DashScope text-embedding-v1（1536维向量）
  - 向量存储：FAISS（内存索引 + 磁盘持久化）
  - 检索策略：向量检索 + BM25 关键词检索 → RRF 融合排序
  - LLM：通义千问 qwen-plus-2025-07-28（OpenAI 兼容接口）

使用方式：
  python main.py
  启动后进入交互式问答界面，输入课程相关问题即可获取回答。
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Generator, Union

# 将项目根目录 cs_rag/ 加入 sys.path，使 config.key 可被导入
sys.path.insert(0, str(Path(__file__).parent.parent))
# 将 C9 目录加入 sys.path，使 rag_config 和 rag_modules 可被导入
sys.path.insert(0, str(Path(__file__).parent))

from rag_config import DEFAULT_CONFIG, RAGConfig
from config.key import qianwen_key
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)

# 设置 DashScope API Key 环境变量（供 DashScopeEmbeddings 使用）
os.environ.setdefault("DASHSCOPE_API_KEY", qianwen_key)

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CourseRAGSystem:
    """
    CS课程RAG系统主类

    工作流程：
      1. 初始化模块 → 2. 构建知识库（加载文档/分块/向量化/保存索引）
      → 3. 用户提问 → 4. 查询路由/重写 → 5. 混合检索 → 6. LLM生成回答
    """

    def __init__(self, config: RAGConfig = None):
        """
        初始化 RAG 系统

        Args:
            config: RAGConfig 配置对象，默认使用 DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module: DataPreparationModule = None
        self.index_module: IndexConstructionModule = None
        self.retrieval_module: RetrievalOptimizationModule = None
        self.generation_module: GenerationIntegrationModule = None

        # 校验数据文件是否存在
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"课程数据文件不存在: {data_path.resolve()}\n"
                f"请先运行项目根目录的脚本生成 cs_course_rag_cards.md"
            )

        # 校验 API Key
        if not qianwen_key or qianwen_key.startswith("your-"):
            raise ValueError(
                "请先在 config/key.py 中设置有效的 DashScope API Key (qianwen_key)"
            )

    # ==================================================================
    # 系统初始化与知识库构建
    # ==================================================================

    def initialize_system(self):
        """
        初始化所有子模块

        模块初始化顺序：
          1. 数据准备模块 —— 负责加载和分块课程文档
          2. 索引构建模块 —— 负责向量化和 FAISS 索引管理
          3. 生成集成模块 —— 负责 LLM 初始化与回答生成
        """
        print("正在初始化 CS 课程 RAG 系统...")

        # 1. 数据准备模块 —— 不需要额外参数，data_path 在 load_documents() 时使用
        print("  初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 索引构建模块 —— 使用 DashScope text-embedding-v1 嵌入模型
        print("  初始化索引构建模块（DashScope Embedding）...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )

        # 3. 生成集成模块 —— 使用千问大模型
        print("  初始化生成集成模块（千问 LLM）...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        print("系统初始化完成！")

    def build_knowledge_base(self):
        """
        构建知识库 —— 加载课程文档、分块、构建向量索引

        流程说明：
          1. 尝试加载已持久化的 FAISS 索引（避免重复向量化）
          2. 如果索引不存在，则：加载文档 → 分块 → 向量化 → 保存索引
          3. 无论走哪条路径，都需要加载文档对象供检索模块使用
          4. 初始化检索优化模块
          5. 输出知识库统计信息
        """
        print("\n正在构建课程知识库...")

        # 第一步：尝试加载已有索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            # 索引加载成功 —— 直接复用，仅需加载原始文档供检索模块使用
            print("成功加载已保存的向量索引，无需重复向量化")
            print("加载课程文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            # 索引不存在 —— 走完整构建流程
            print("未找到已保存的索引，开始完整构建流程...")

            # 第二步：加载并解析课程 Markdown 文件
            print("  加载课程文档...")
            self.data_module.load_documents()

            # 第三步：Markdown 结构感知分块（按 H2/H3 标题切分）
            print("  进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 第四步：调用 DashScope API 向量化所有 chunk，构建 FAISS 索引
            print("  构建向量索引（调用 DashScope Embedding API）...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 第五步：持久化保存索引
            print("  保存向量索引到磁盘...")
            self.index_module.save_index()

        # 第六步：初始化检索优化模块（混合检索 + RRF 重排）
        print("  初始化检索优化模块...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 第七步：输出统计信息
        stats = self.data_module.get_statistics()
        print(f"\n知识库统计:")
        print(f"  课程总数: {stats['total_courses']}")
        print(f"  文本块数: {stats['total_chunks']}")
        if "domains" in stats:
            print(f"  领域分布: {stats['domains']}")
        if "difficulties" in stats:
            print(f"  难度分布: {stats['difficulties']}")

        print("知识库构建完成！")

    # ==================================================================
    # 核心问答接口
    # ==================================================================

    def ask_question(
        self, question: str, stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        回答用户问题 —— 完整的 RAG 问答流水线

        流水线步骤：
          1. 查询路由 —— 判断是 list / detail / general 类型
          2. 查询重写 —— 对模糊查询进行智能改写（list 类型跳过）
          3. 混合检索 —— 向量 + BM25 → RRF 重排序
          4. 元数据过滤 —— 从 query 中自动提取领域/难度等过滤条件
          5. LLM 生成 —— 根据路由类型选择不同的回答模板
          6. 返回结果 —— 支持普通输出和流式输出两种模式

        Args:
            question: 用户问题
            stream: 是否使用流式输出（逐字返回）

        Returns:
            生成的回答字符串，或流式输出的生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先调用 build_knowledge_base() 构建知识库")

        print(f"\n用户问题: {question}")

        # 步骤1：查询路由 —— 由 LLM 判断查询属于哪种类型
        route_type = self.generation_module.query_router(question)
        print(f"查询类型: {route_type}")

        # 步骤2：查询重写 —— list 类型保持原样，detail/general 类型进行智能重写
        if route_type == "list":
            rewritten_query = question
            print(f"列表查询保持原样: {question}")
        else:
            print("智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        # 步骤3+4：检索 —— 优先尝试元数据过滤，否则走混合检索
        print("检索相关课程文档...")
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"  应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query, filters, top_k=self.config.top_k
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query, top_k=self.config.top_k
            )

        # 打印检索到的课程名称，帮助用户理解检索结果
        if relevant_chunks:
            course_names = []
            for chunk in relevant_chunks:
                name = chunk.metadata.get("course_name", "未知")
                if name not in course_names:
                    course_names.append(name)
            print(f"找到 {len(relevant_chunks)} 个相关文档块，涉及课程: {', '.join(course_names)}")
        else:
            print("未找到相关文档块")

        # 如果没有检索到任何内容，直接返回
        if not relevant_chunks:
            return "抱歉，没有找到相关的课程信息。请尝试其他关键词或更宽泛的查询。"

        # 步骤5：根据路由类型选择回答策略
        if route_type == "list":
            # 列表查询 —— 获取父文档（完整课程），生成课程列表
            print("生成课程列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            return self.generation_module.generate_list_answer(question, relevant_docs)
        elif route_type == "detail":
            # 详细查询 —— 获取父文档，生成分模块详细回答
            print("获取完整课程文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            print("生成详细回答...")
            if stream:
                return self.generation_module.generate_detail_answer_stream(
                    question, relevant_docs
                )
            else:
                return self.generation_module.generate_detail_answer(
                    question, relevant_docs
                )
        else:
            # 一般查询 —— 获取父文档，生成基础回答
            print("获取完整课程文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            print("生成回答...")
            if stream:
                return self.generation_module.generate_basic_answer_stream(
                    question, relevant_docs
                )
            else:
                return self.generation_module.generate_basic_answer(
                    question, relevant_docs
                )

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户查询中自动提取元数据过滤条件

        支持的过滤维度：
          - 领域（domain）：AI、系统、软件工程、数据科学、网络安全 等
          - 难度（difficulty）：★☆☆☆☆ ~ ★★★★★

        匹配逻辑：
          遍历知识库中已知的领域和难度标签，检查是否出现在查询字符串中。
          优先匹配更长的标签（避免"系统"误匹配到"操作系统"）。

        Args:
            query: 用户查询文本

        Returns:
            元数据过滤条件字典，如 {"domain": "AI", "difficulty": "★★★☆☆"}
            如果没有匹配到任何条件，返回空字典 {}
        """
        filters = {}

        # 领域关键词匹配
        # 注意：排序按关键词长度降序，避免短词误匹配
        # 例如 "网络安全" 应优先于 "网络" 匹配
        domain_keywords = sorted(
            DataPreparationModule.DOMAIN_LABELS,
            key=len,
            reverse=True,
        )
        for domain in domain_keywords:
            if domain in query:
                filters["domain"] = domain
                break

        # 难度关键词匹配
        difficulty_keywords = sorted(
            DataPreparationModule.DIFFICULTY_LABELS,
            key=len,
            reverse=True,
        )
        for diff in difficulty_keywords:
            if diff in query:
                filters["difficulty"] = diff
                break

        return filters

    # ==================================================================
    # 便捷查询方法
    # ==================================================================

    def search_by_domain(self, domain: str, query: str = "") -> List[str]:
        """
        按领域搜索课程

        Args:
            domain: 领域名称（如 "AI"、"系统"、"网络安全"）
            query: 可选的额外查询条件

        Returns:
            匹配的课程名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先调用 build_knowledge_base() 构建知识库")

        search_query = query if query else f"{domain}方向课程"
        filters = {"domain": domain}

        docs = self.retrieval_module.metadata_filtered_search(
            search_query, filters, top_k=10
        )

        course_names = []
        for doc in docs:
            name = doc.metadata.get("course_name", "")
            if name and name not in course_names:
                course_names.append(name)

        return course_names

    def get_course_overview(self, course_name: str) -> str:
        """
        获取指定课程的概述信息

        Args:
            course_name: 课程名称（如 "数据结构"、"操作系统原理"）

        Returns:
            课程概述回答
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先调用 build_knowledge_base() 构建知识库")

        docs = self.retrieval_module.hybrid_search(course_name, top_k=3)
        return self.generation_module.generate_detail_answer(
            f"{course_name}的课程概述、核心内容和学习建议", docs
        )

    # ==================================================================
    # 交互式问答
    # ==================================================================

    def run_interactive(self):
        """
        启动交互式问答循环

        使用说明：
          - 输入课程相关问题，系统检索知识库并生成回答
          - 支持课程查询、领域筛选、学习路线咨询等多种问题类型
          - 输入"退出" 或 Ctrl+C 结束程序
          - 可以选择流式输出（逐字打印）或普通输出

        示例问题：
          - "推荐几门AI方向的课程"
          - "数据结构讲什么内容"
          - "机器学习和深度学习有什么关系"
          - "操作系统原理的面试考点有哪些"
        """
        print("=" * 60)
        print("  CS 课程知识库 RAG 系统 —— 交互式问答")
        print("=" * 60)
        print("基于 22 门计算机专业核心课程知识卡片构建")
        print("输入'退出'或 Ctrl+C 结束程序")
        print("输入'帮助'查看示例问题")

        # 步骤1：初始化所有子模块
        self.initialize_system()

        # 步骤2：构建知识库（首次运行会调用 API 向量化，后续复用缓存）
        self.build_knowledge_base()

        print("\n交互式问答开始！")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()

                if user_input.lower() in ("退出", "quit", "exit", ""):
                    print("再见！")
                    break

                if user_input.lower() in ("帮助", "help"):
                    self._print_help()
                    continue

                # 询问输出模式
                stream_choice = (
                    input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                )
                use_stream = stream_choice != "n"

                print("\n回答:")
                if use_stream:
                    # 流式输出 —— 逐字打印，用户体验更好
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出 —— 一次性返回完整回答
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                logger.error("处理问题时出错: %s", e, exc_info=True)
                print(f"处理问题时出错: {e}")

    def _print_help(self):
        """打印帮助信息"""
        print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  示例问题：
    📋 课程推荐类：
       - 推荐几门AI方向的课程
       - 有哪些系统方向的课
       - 列出数据科学相关的课程
       - ★★★☆☆ 难度的课程有哪些

    📖 课程详情类：
       - 数据结构讲什么内容
       - 操作系统的学习路线是什么样的
       - 机器学习需要什么前置知识
       - 编译原理的面试高频考点有哪些

    🔗 课程关系类：
       - 机器学习和深度学习有什么关系
       - AI方向应该按什么顺序学
       - 计算机网络和分布式系统的联系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)


# ==================================================================
# 程序入口
# ==================================================================

def main():
    """主函数 —— 创建 RAG 系统并启动交互式问答"""
    try:
        # 设置工作目录为 C9 目录（确保相对路径正确）
        os.chdir(Path(__file__).parent)

        rag_system = CourseRAGSystem()
        rag_system.run_interactive()

    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error("系统运行出错: %s", e, exc_info=True)
        print(f"系统错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
