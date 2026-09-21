
"""RAG 检索总结服务。

用户提问后，先从 Chroma 检索相关参考资料，再把“原问题 + 参考资料”填入
专用 Prompt，最后交给聊天模型总结。这个服务本身不是 Agent，而是被 Agent
作为 ``rag_summarize`` 工具调用的一条确定性处理链。
"""

# Document 是 LangChain 的标准资料对象：page_content 保存正文，metadata 保存来源信息。
from langchain_core.documents import Document
# StrOutputParser 将 ChatModel 返回的 AIMessage 转成普通字符串，便于工具直接返回文本。
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
# PromptTemplate 负责检查模板变量，并在 invoke 时把 input/context 填入 Prompt。
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


def print_prompt(prompt):
    """把渲染后的完整 RAG Prompt 打印到终端，并原样传给下一环节。

    该函数插入 LCEL 管道主要用于教学调试，不改变 Prompt 内容。正式环境中
    Prompt 可能含用户数据或私有资料，不应无条件完整打印。

    Args:
        prompt: PromptTemplate 渲染后生成的 PromptValue。

    Returns:
        原始 prompt，使后续聊天模型仍能接收到它。
    """
    # 在 LCEL 管道中，普通可调用函数会被自动适配成 RunnableLambda。
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    """封装“检索 → 拼上下文 → 模型总结”的完整 RAG 流程。"""

    def __init__(self):
        """初始化向量检索器、Prompt、聊天模型和 LCEL Chain。"""
        # VectorStoreService 连接持久化的 Chroma 数据库。
        self.vector_store = VectorStoreService()
        # 创建连接不会调用 load_document；知识文件需要先通过向量库模块的入库入口加载。

        # get_retriever() 无入参；返回的 Retriever 保存到本文件 self.retriever。
        # Retriever 统一了检索接口；当前 search_kwargs 中的 k=3 来自配置。
        self.retriever = self.vector_store.get_retriever()

        # load_rag_prompts() 无入参，返回模板字符串；该返回值随后作为实参
        # self.prompt_text → PromptTemplate.from_template(template) 的形参 template。
        # 从 prompts/rag_summarize.txt 读取含 {input} 和 {context} 的模板文本。
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 模板文本在本服务初始化时读取并保存；修改提示词文件后，已有 chain 不会自动更新。

        # RAG 内部总结与外层 Agent 当前共用同一个 chat_model 对象。
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        """使用 LCEL 语法组装并返回 RAG 处理链。

        管道顺序：
        输入字典 → PromptTemplate → 打印调试 → ChatModel → 纯字符串。
        ``StrOutputParser`` 会从模型返回的 AIMessage 中提取正文。
        """
        # LCEL 的 | 表示把左侧输出交给右侧输入，数据依次经历模板渲染、调试打印、
        # 模型生成和正文提取。这里得到的是可复用 Runnable，尚未执行远程请求。
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """只做语义检索，不调用聊天模型总结。

        Args:
            query: 用户问题或检索词。

        Returns:
            按相关性返回的 Document 列表；当前通常最多 3 个。

        单独调用本方法，可以区分“检索阶段没找到好资料”和“模型总结不理想”。
        """
        # 传参：本方法形参 query → Retriever.invoke(input, ...) 的形参 input；
        # Retriever 再把该查询文本交给 model/factory.py 创建的 embed_model 做查询向量化。
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """检索资料，并生成严格基于资料的中文总结。

        Args:
            query: 用户原始问题或由 Agent 整理的检索词。

        Returns:
            RAG Chain 输出的纯文本答案。
        """

        # 本文件内传参：rag_summarize(query) 的形参 query →
        # retriever_docs(self, query) 的同名形参 query。
        context_docs = self.retriever_docs(query)
        # 返回片段表示向量相似度较高，不代表资料一定能回答问题；当前未设置相关性阈值。

        # 将多个 Document 拼成一个带编号的上下文字符串，同时保留 metadata。
        # metadata 常包含来源文件和 PDF 页码，有助于理解答案来自哪里。
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        # 没有检索结果时 context 仍为空字符串，下面依然会调用模型；代码未提供空结果分支。
        # metadata 直接转成文本供模型参考，此处不生成可点击引用，也不校验回答是否忠于资料。

        # 传给 LCEL Chain 的映射：实参字典整体 → self.chain.invoke(input) 的形参 input；
        # 键 "input" 的值 query → PromptTemplate 的 {input}，
        # 键 "context" 的值 context → PromptTemplate 的 {context}。
        # 注意外层 invoke 的形参 input 与模板中的变量名 {input} 恰好同名，但层级不同。
        # invoke 是同步调用，会等待模型生成完整总结后返回。
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    # 独立测试 RAG，不经过 Agent 和 Streamlit。
    # 推荐从项目根目录执行：python -m rag.rag_service
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
