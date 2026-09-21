"""统一加载项目中的 YAML 配置文件。

YAML 使用 ``key: value`` 形式保存易变参数。把模型名、文件路径和分片参数放在
配置中，可以避免为了调整参数而修改业务代码。

本模块在 import 时立即读取四份配置并创建全局字典；因此配置文件不存在、
YAML 格式错误或编码错误都会在程序启动早期出现。
"""

import yaml

from utils.path_tool import get_abs_path


# 默认值传参："config/rag.yml" → utils/path_tool.py 中
# get_abs_path(relative_path) 的形参 relative_path；其返回值再成为 config_path 的默认值。
def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    """读取聊天模型与 Embedding 模型配置。

    Args:
        config_path: 配置文件绝对路径，默认指向项目 ``config/rag.yml``。
        encoding: 文件编码，项目配置统一使用 UTF-8。
    """
    # 默认参数会在函数定义（模块导入）时求值，而不是每次调用时重新求值。
    # 调用者也可以传入另一份配置路径，便于开发环境和测试环境使用不同配置。
    with open(config_path, "r", encoding=encoding) as f:
        # FullLoader 会把 YAML 映射解析为 Python dict，并解析常规标量类型。
        # 例如数字会成为 int，列表会成为 list；本函数没有校验必填键和数据类型。
        return yaml.load(f, Loader=yaml.FullLoader)


# 默认值传参："config/chroma.yml" → get_abs_path(relative_path)；返回值成为 config_path 默认值。
def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    """读取 Chroma、Retriever 和文本分片参数。

    这些配置控制集合名称、持久化目录、召回数量 k、知识文件类型，以及切片大小、
    重叠长度和分隔符。返回值供 ``VectorStoreService`` 初始化时直接使用。
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# 默认值传参："config/prompts.yml" → get_abs_path(relative_path)；返回值成为 config_path 默认值。
def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    """读取三类 Prompt 文件的相对路径。

    这里读取的是“Prompt 文件在哪里”，不是 Prompt 正文。正文由
    ``prompt_loader.py`` 根据这些路径在需要时读取。
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# 默认值传参："config/agent.yml" → get_abs_path(relative_path)；返回值成为 config_path 默认值。
def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    """读取 Agent 业务数据配置，例如模拟外部 CSV 路径。

    把业务数据位置留在配置层，可以更换文件而不改工具函数；当前只有
    ``external_data_path``，以后也可扩展服务地址、超时等 Agent 业务参数。
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# 模块级加载：其他文件只需 import 这些字典，不必重复读取 YAML。
# 对应代价是运行中修改 YAML 不会自动刷新，通常需要重启 Python/Streamlit 进程。
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
# 以上四个名称分别服务于“模型”“向量库”“提示词路径”“Agent 业务数据”。
# 它们是普通可变字典，代码没有设置只读保护；运行时修改会影响后续读取者，
# 但已经用旧配置创建好的模型、向量库或 Prompt 对象不会自动重新初始化。


if __name__ == '__main__':
    # 直接运行本文件时，用一个配置项做最小读取测试。
    print(rag_conf["chat_model_name"])
