"""知识文件读取与文件指纹工具。

VectorStoreService 使用这里的函数发现 TXT/PDF、读取为 LangChain Document，
并用 MD5 判断完全相同的文件内容是否已经入库。
"""

import os
import hashlib

from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# Loader 的意义是把不同文件格式统一转换成 Document。上层切分器和向量库只需
# 面对 Document，不需要分别理解 PDF 页结构或文本文件编码。


def get_file_md5_hex(filepath: str):
    """计算文件内容的 MD5 十六进制指纹。

    Args:
        filepath: 待计算的文件路径。

    Returns:
        成功时返回 32 位十六进制字符串；路径无效或读取失败时返回 None。

    MD5 在这里用于内容去重，不用于密码或安全签名。
    """

    # 先区分“不存在”和“不是普通文件”，便于日志给出更准确的原因。
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    # hashlib.md5() 创建增量摘要对象；可多次 update，而不必把整份文件读入内存。
    md5_obj = hashlib.md5()

    # 每次只读取 4 KB，避免大文件一次性进入内存。
    chunk_size = 4096
    try:
        # 必须按二进制读取，文本编码与换行转换不应影响文件指纹。
        with open(filepath, "rb") as f:
            # 海象运算符 := 同时完成读取和赋值；读到 b"" 时循环结束。
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
                # update 按顺序吸收每一块字节；块大小只影响读取效率，不改变最终摘要。

            # 下方保留的注释展示了不使用海象运算符时的等价传统写法。
            """
            chunk = f.read(chunk_size)
            while chunk:
                
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            # hexdigest 将 16 字节摘要表示为便于写入文本文件和比较的 32 位十六进制串。
            return md5_hex
    except Exception as e:
        # 读取权限、文件占用或磁盘错误等都会进入这里。
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """返回目录第一层中后缀符合要求的文件路径。

    Args:
        path: 要扫描的目录。
        allowed_types: 可接受的后缀元组，例如 ``("txt", "pdf")``。

    Returns:
        正常时返回文件路径元组。当前源码在 path 不是目录时误返回
        ``allowed_types``，这是文档中已指出的示例缺陷，调用时需要留意。

    本函数不递归子目录，且当前后缀判断区分大小写。
    """
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    # os.listdir 仅返回当前目录中的名称，顺序由文件系统决定，且包含子目录名称。
    for f in os.listdir(path):
        # str.endswith 接受后缀元组；例如 name.txt 会匹配 "txt"。
        # 这里没有额外调用 isfile，因此名称恰好以允许后缀结尾的目录也可能被加入。
        if f.endswith(allowed_types):
            # join 把目录与文件名组合成可供 Loader 直接使用的完整路径。
            files.append(os.path.join(path, f))

    # 返回 tuple 表示调用方通常只遍历结果，不需要修改列表。
    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """读取 PDF，并按 Loader 规则返回一个或多个 Document。

    PDF 每页通常成为一个 Document，metadata 中会保留来源与页码。扫描图片型
    PDF 没有文本层时需要先做 OCR。
    """
    # passwd 用于受密码保护的 PDF；None 表示不提供密码。
    # 构造 Loader 只保存参数，load() 才会打开文件并提取文本。
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    """以 UTF-8 读取纯文本并返回 Document 列表。

    显式指定 UTF-8 可避免 Windows 默认 GBK 与资料编码不一致造成的错误。
    """
    # TextLoader 通常把整份 TXT 作为一个 Document；后续再由文本切分器拆成片段。
    return TextLoader(filepath, encoding="utf-8").load()
