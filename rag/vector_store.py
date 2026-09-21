"""本地知识库的构建与检索服务。

本模块负责两条路径：

1. 入库：TXT/PDF → Document → 文本分片 → Embedding → Chroma；
2. 查询：用户问题 → Embedding → Chroma 相似度搜索 → Document 列表。

``md5.text`` 只用于避免完全相同的文件重复入库，不是向量数据库本身。
"""

# Chroma 是向量数据库组件，负责持久化文本、向量和 metadata，并执行相似度搜索。
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
# RecursiveCharacterTextSplitter 尽量沿段落和句子边界切片，比固定字符硬切更易保留语义。
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:
    """封装 Chroma 连接、文本分片、文档入库和 Retriever 创建。"""

    def __init__(self):
        """按 YAML 配置连接 Chroma，并创建递归字符切分器。"""
        # Chroma 在 persist_directory 指定的目录中持久保存集合数据。
        # 注意：当前 persist_directory 直接使用配置值 "chroma_db"，没有转换成
        # 项目绝对路径，因此实际位置取决于启动进程时的工作目录。
        self.vector_store = Chroma(
            # 配置值 → Chroma.__init__(...) 的同名形参 collection_name。
            collection_name=chroma_conf["collection_name"],
            # model/factory.py 的 embed_model → Chroma 的形参 embedding_function。
            embedding_function=embed_model,
            # 配置值 → Chroma 的形参 persist_directory。
            persist_directory=chroma_conf["persist_directory"],
        )
        # MD5 文件与 Chroma 目录独立保存：只清空向量库而保留指纹记录时，旧文件仍会
        # 被判断为已入库。更换集合或重建库时，需要让指纹记录与目标集合保持一致。

        # RecursiveCharacterTextSplitter 会按 separators 给出的优先顺序寻找合适
        # 边界；若大段文本仍然过长，则逐步尝试更细的分隔符。
        self.spliter = RecursiveCharacterTextSplitter(
            # 以下三个配置值分别传给构造函数的同名形参。
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            # len 按 Python Unicode 字符数计算长度，不等同于模型 token 数。
            length_function=len,
        )
        # spliter 仅保存切分规则，不读取文件；真正切分发生在 split_documents 调用时。

    def get_retriever(self):
        """把 Chroma 包装为统一 Retriever 接口。

        Returns:
            可通过 ``invoke(query)`` 检索 Document 的 VectorStoreRetriever。
            ``k`` 表示每次最多取回多少个相似片段。
        """
        # 实参字典 → Chroma.as_retriever(..., search_kwargs) 的形参 search_kwargs；
        # 其中配置值 chroma_conf["k"] → 检索参数 k，表示最多返回的片段数。
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})
        # 本方法只创建检索器，不触发检索；实际查询发生在调用方的 invoke(query) 中。

    def load_document(self):
        """扫描知识目录，把尚未处理的 TXT/PDF 切分并写入向量库。

        每个文件成功写入后，将其 MD5 追加到配置的指纹文件。下次运行时，
        相同内容的文件会被跳过。单个文件失败只记录日志并继续处理其他文件。

        Returns:
            None。
        """

        def check_md5_hex(md5_for_check: str):
            """检查给定 MD5 是否已经记录。

            指纹文件不存在时先创建空文件，并返回 False，表示尚未处理。
            """
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建空指纹文件。这里只创建文件，父目录应已存在。
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                # 逐行精确比较摘要；文件较大时是线性查找，样例数据规模下足够简单直观。
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check: str):
            """在文档成功写入向量库后，追加保存它的 MD5 指纹。"""
            # "a" 是追加模式，不会覆盖之前文件的摘要。
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            """根据文件后缀选择 Loader，并统一返回 Document 列表。"""
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        # data_path 通过 get_abs_path 固定为项目根目录下的 data，
        # listdir_with_allowed_type 只扫描该目录第一层，不会递归 external 子目录。
        allowed_files_path: list[str] = listdir_with_allowed_type(
            # get_abs_path 的返回值 → utils/file_handler.py 中
            # listdir_with_allowed_type(path, ...) 的形参 path。
            get_abs_path(chroma_conf["data_path"]),
            # tuple(...) 的返回值 → listdir_with_allowed_type(..., allowed_types)
            # 的形参 allowed_types。
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 跨文件传参：循环变量 path → utils/file_handler.py 中
            # get_file_md5_hex(filepath) 的形参 filepath。
            md5_hex = get_file_md5_hex(path)

            # get_file_md5_hex 失败时可能返回 None；当前代码仍会拿 None 去指纹表比较，
            # 随后还可能尝试加载原路径，最终由下方异常处理记录失败。
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                # 本文件内传参：循环变量 path → get_file_documents(read_path) 的形参 read_path；
                # 后者再把 read_path 传给 utils/file_handler.py 的 txt_loader(filepath)
                # 或 pdf_loader(filepath) 的形参 filepath。
                # Loader 把每个源文件转换成一个或多个 LangChain Document；
                # Document 包含 page_content 正文和 metadata 来源信息。
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # 实参 documents → RecursiveCharacterTextSplitter.split_documents(documents)
                # 的形参 documents。
                split_document: list[Document] = self.spliter.split_documents(documents)
                # split_documents 会复制来源 metadata 到每个片段，使检索结果仍可追溯到源文件/页码。

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 实参 split_document → Chroma.add_documents(documents, ...) 的形参 documents。
                # add_documents 会调用 Embedding 模型向量化每个片段，然后把文本、
                # 向量和 metadata 一起保存到 Chroma。这一步可能发生外部 API 调用。
                self.vector_store.add_documents(split_document)

                # 本文件内传参：md5_hex → save_md5_hex(md5_for_check) 的形参 md5_for_check。
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info=True 会把完整调用堆栈写入日志，便于定位具体 Loader、
                # Embedding 或 Chroma 环节。continue 让其他文件仍可继续入库。
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    # 独立运行时先增量入库，再以“迷路”为测试词验证 Retriever。
    # 推荐从项目根目录执行：python -m rag.vector_store
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    # invoke 会对查询进行 Embedding，并返回配置 k 指定数量的相似 Document。
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)


