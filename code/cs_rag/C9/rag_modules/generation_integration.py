"""
生成集成模块 —— 使用千问大模型 + LangChain LCEL 链式调用
"""

import os
import logging
from typing import List, Generator

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 从项目根目录的 config.key 模块读取 API Key
from config.key import qianwen_key

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 —— 负责LLM初始化、查询路由、查询重写与回答生成"""

    def __init__(
        self,
        model_name: str = "qwen-plus-2025-07-28",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        初始化生成集成模块

        Args:
            model_name: 千问模型名称，默认 qwen-plus-2025-07-28
            temperature: 生成温度，越低越确定（课程问答需要准确性，默认0.1）
            max_tokens: 单次生成的最大 token 数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm: ChatOpenAI = None
        self._setup_llm()

    def _setup_llm(self):
        """
        初始化千问大语言模型

        使用 OpenAI 兼容接口调用阿里云 DashScope 上的千问模型：
        - base_url 指向阿里云兼容端点
        - api_key 来自 config/key.py 中的 qianwen_key
        """
        logger.info("正在初始化千问 LLM: %s", self.model_name)

        # 设置 DashScope API Key 环境变量（DashScopeEmbeddings 也会用到）
        os.environ.setdefault("DASHSCOPE_API_KEY", qianwen_key)

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=qianwen_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        logger.info("千问 LLM 初始化完成")

    # ==================================================================
    # 查询分析 —— 路由 + 重写
    # ==================================================================

    def query_router(self, query: str) -> str:
        """
        查询路由 —— 将用户问题分类为三种类型，选择不同的回答策略

        三种路由类型：
          - 'list'   ：用户想获取课程列表/推荐（只需课程名）
          - 'detail' ：用户想了解某门课程的具体内容（详细回答）
          - 'general'：一般性问题（综合多个课程的信息）

        Args:
            query: 用户原始查询

        Returns:
            路由类型字符串：'list' / 'detail' / 'general'
        """
        prompt = ChatPromptTemplate.from_template("""
你是一个计算机课程问答的分类器。请将用户问题分为以下三类之一：

1. 'list' - 用户想要获取课程列表或推荐，只需要课程名称
   例如：推荐几门AI方向的课程、有哪些系统方向的课、列出数据科学的课程

2. 'detail' - 用户想深入了解某门具体课程的内容
   例如：数据结构讲什么、操作系统的学习路线、机器学习的前置课程是什么

3. 'general' - 一般性问题，可能涉及多门课程的比较或综合信息
   例如：AI方向怎么规划学习路径、计算机网络和分布式系统有什么关系

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()
        if result in ("list", "detail", "general"):
            return result
        return "general"  # 兜底默认

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 —— 将模糊或口语化的查询改写为更精确的检索查询

        改写原则：
          - 具体明确的课程查询（如"数据结构讲什么"）保持原样
          - 模糊宽泛的查询（如"学什么好"、"有哪些课"）增加领域/难度限定
          - 补充同义词和专业术语，提升检索命中率

        Args:
            query: 用户原始查询

        Returns:
            重写后的查询（或原查询，如果无需改写）
        """
        prompt = PromptTemplate(
            template="""
你是一个计算机课程查询优化助手。分析用户的查询，判断是否需要重写以提高课程搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体课程名称：如"数据结构的主要内容"、"操作系统学习路线"
   - 明确的课程询问：如"机器学习需要什么前置知识"、"编译原理的面试考点"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"学什么"、"有什么课程"、"推荐个方向"
   - 缺乏具体信息：如"AI"、"系统方向"、"简单的课"
   - 口语化表达：如"想学编程"、"找工作的课"

重写原则：
- 保持原意不变
- 增加课程相关术语
- 保持简洁，不超过30字

示例：
- "学什么" → "计算机专业核心课程推荐"
- "AI方向" → "人工智能方向课程体系与学习路线"
- "数据结构讲什么" → "数据结构讲什么"（保持原查询）
- "想学编程" → "程序设计入门课程推荐"

请输出最终查询（如果不需要重写就原样返回）:""",
            input_variables=["query"],
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()
        if response != query:
            logger.info("查询已重写: '%s' → '%s'", query, response)
        else:
            logger.info("查询无需重写: '%s'", query)
        return response

    # ==================================================================
    # 回答生成
    # ==================================================================

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        基础回答 —— 根据检索到的课程文档回答用户问题

        Args:
            query: 用户查询
            context_docs: 检索到的父文档列表（完整课程内容）

        Returns:
            LLM 生成的回答文本
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位资深的计算机科学教育专家和课程顾问。请根据以下课程知识卡片信息，回答用户的问题。

用户问题: {question}

相关课程信息:
{context}

要求：
- 回答准确、专业、简洁，面向计算机专业学生
- 优先使用课程知识卡片中的原文信息
- 如果涉及多门课程，说明它们之间的关系
- 如果信息不足，请诚实说明，不要编造

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(query)

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        列表式回答 —— 适用于"推荐课程"类查询

        Args:
            query: 用户查询
            context_docs: 检索到的父文档列表

        Returns:
            结构化的课程列表回答
        """
        if not context_docs:
            return "抱歉，没有找到匹配的课程信息。请尝试其他关键词或领域。"
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位计算机课程顾问。用户想要获取课程列表或推荐，请根据以下课程信息生成清晰有序的回答。

用户问题: {question}

相关课程信息:
{context}

要求：
- 以列表形式列出课程名称
- 每门课程附一句话简介（课程简介原文或你的概括）
- 标注每门课程的难度和求职重要程度
- 如果课程数量较多，按推荐优先级或难度递进排列

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(query)

    def generate_detail_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        详细回答 —— 适用于深入了解某门课程

        Args:
            query: 用户查询
            context_docs: 检索到的父文档列表

        Returns:
            详细的课程介绍回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位资深的计算机科学教育专家。用户想深入了解某门课程，请根据课程知识卡片提供详尽的回答。

用户问题: {question}

相关课程信息:
{context}

请灵活组织回答，建议包含以下内容（可根据实际情况增减）：
1. **课程概述**：课程定位与核心内容
2. **难度与前置**：适合什么阶段学习，需要哪些先修知识
3. **核心知识点**：最重要的知识模块
4. **学习路线建议**：分阶段的学习计划
5. **实践与工具**：推荐的项目和工具
6. **面试与求职**：面试高频考点和求职方向

注意：
- 优先使用知识卡片原文中的信息
- 如果卡片中没有的信息，可以基于你的知识补充，但需标注"补充建议"
- 内容要有条理，便于学生阅读理解

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke(query)

    # ==================================================================
    # 流式输出版本
    # ==================================================================

    def generate_basic_answer_stream(
        self, query: str, context_docs: List[Document]
    ) -> Generator[str, None, None]:
        """基础回答的流式输出版本"""
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是一位资深的计算机科学教育专家和课程顾问。请根据以下课程知识卡片信息，回答用户的问题。

用户问题: {question}

相关课程信息:
{context}

要求：
- 回答准确、专业、简洁，面向计算机专业学生
- 优先使用课程知识卡片中的原文信息
- 如果涉及多门课程，说明它们之间的关系
- 如果信息不足，请诚实说明，不要编造

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    def generate_detail_answer_stream(
        self, query: str, context_docs: List[Document]
    ) -> Generator[str, None, None]:
        """详细回答的流式输出版本"""
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
你是一位资深的计算机科学教育专家。用户想深入了解某门课程，请根据课程知识卡片提供详尽的回答。

用户问题: {question}

相关课程信息:
{context}

请灵活组织回答，建议包含以下内容（可根据实际情况增减）：
1. **课程概述**：课程定位与核心内容
2. **难度与前置**：适合什么阶段学习，需要哪些先修知识
3. **核心知识点**：最重要的知识模块
4. **学习路线建议**：分阶段的学习计划
5. **实践与工具**：推荐的项目和工具
6. **面试与求职**：面试高频考点和求职方向

注意：
- 优先使用知识卡片原文中的信息
- 如果卡片中没有的信息，可以基于你的知识补充，但需标注"补充建议"
- 内容要有条理，便于学生阅读理解

回答:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        for chunk in chain.stream(query):
            yield chunk

    # ==================================================================
    # 内部工具
    # ==================================================================

    def _build_context(self, docs: List[Document], max_length: int = 3000) -> str:
        """
        将检索到的文档列表拼接为 LLM 可接受的上下文字符串

        每个文档会附带课程名称、领域、难度等元数据头部信息，
        并控制总长度不超过 max_length 字符，避免超出模型窗口。

        Args:
            docs: 文档列表
            max_length: 上下文最大字符数

        Returns:
            格式化后的上下文字符串
        """
        if not docs:
            return "暂无相关课程信息。"

        parts: List[str] = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            name = doc.metadata.get("course_name", "未知课程")
            domain = doc.metadata.get("domain", "")
            difficulty = doc.metadata.get("difficulty", "")
            importance = doc.metadata.get("job_importance", "")

            # 构建课程元数据头部，便于 LLM 快速定位
            header = f"【课程 {i}】{name}"
            if domain:
                header += f" | 领域: {domain}"
            if difficulty:
                header += f" | 难度: {difficulty}"
            if importance:
                header += f" | 求职重要度: {importance}"

            doc_text = f"{header}\n{doc.page_content}\n"

            if current_length + len(doc_text) > max_length:
                break

            parts.append(doc_text)
            current_length += len(doc_text)

        divider = "\n" + "=" * 50 + "\n"
        return divider.join(parts)
