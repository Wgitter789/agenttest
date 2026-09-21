"""集中创建项目使用的聊天模型与 Embedding 模型。

聊天模型负责自然语言理解、工具选择和答案生成；Embedding 模型负责把文本
转换成向量，供 Chroma 做语义相似度检索。两者用途不同，但都通过 DashScope
调用阿里云模型，通常依赖环境变量 ``DASHSCOPE_API_KEY``。
"""

from abc import ABC, abstractmethod
from typing import Optional

# Embeddings 是 LangChain 对“文本向量模型”的统一接口，主要提供
# embed_documents（批量文档向量化）和 embed_query（查询向量化）等能力。
from langchain_core.embeddings import Embeddings
# BaseChatModel 是聊天模型的公共基类，用于类型标注；真正实例由 ChatTongyi 提供。
from langchain_community.chat_models.tongyi import BaseChatModel
# DashScopeEmbeddings 将 LangChain 的 Embeddings 接口适配到阿里云百炼向量服务。
from langchain_community.embeddings import DashScopeEmbeddings
# ChatTongyi 将通义千问包装成 LangChain ChatModel，使其可参与 Chain 和 Agent。
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    """模型工厂的抽象接口。

    ABC 表示抽象基类；子类必须实现 ``generator``，从而让调用者使用统一方法
    创建不同类型或不同厂商的模型。
    """

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """创建模型实例；具体类型由子类决定。

        ``@abstractmethod`` 表示该方法只定义统一协议，不能把
        ``BaseModelFactory`` 当成完整工厂直接实例化。返回类型中的 ``|`` 表示
        联合类型；``Optional[T]`` 又允许返回 None。当前两个实现实际都会返回实例，
        因而这个类型声明比真实行为更宽松。
        """
        # pass 是抽象方法的空实现；子类覆盖该方法后，这一行不会参与实际创建过程。
        pass


class ChatModelFactory(BaseModelFactory):
    """创建通义千问聊天模型。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """按 YAML 中的模型名创建并返回 ``ChatTongyi`` 实例。

        ChatModel 接收消息列表并返回 AIMessage；在 Agent 场景中，它还可能返回
        tool_calls，让执行图知道需要调用哪个工具。创建对象不等于立即请求模型，
        真正的远程调用发生在 ``invoke``、``stream`` 或 Agent 执行期间。
        """
        # 配置实参 rag_conf["chat_model_name"] → ChatTongyi.__init__(..., model) 的形参 model。
        # ChatTongyi 实现 LangChain 标准 ChatModel 接口，可被 create_agent 使用。
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    """创建 DashScope 文本向量模型。"""

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """按 YAML 中的模型名创建并返回 ``DashScopeEmbeddings`` 实例。

        Embedding 的输出是浮点数向量，不负责生成自然语言。入库时向量表示知识
        片段，检索时向量表示问题；Chroma 根据两类向量之间的距离寻找相似内容。
        """
        # 配置实参 rag_conf["embedding_model_name"] →
        # DashScopeEmbeddings.__init__(..., model) 的形参 model。
        # 同一向量库的建库和查询必须使用兼容的同一个 Embedding 模型。
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 这两个是模块级单例式对象：首次 import model.factory 时立即创建，之后其他
# 模块通过 import 复用。它们不是等到第一次用户提问时才创建。
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
# 这种“模块导入即初始化”的写法使用方便，但配置错误会在 import 阶段直接暴露；
# 测试时若想替换模型，也需要在引用这些全局对象的模块中进行 Mock 或依赖注入。
