"""根据 YAML 配置读取三类 Prompt 文本。

Prompt 与 Python 代码分离后，可以独立修改角色、工具流程和输出要求。这里的
加载函数统一完成“取配置路径 → 转绝对路径 → UTF-8 读取 → 记录异常”。
"""

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
# 本模块与 config_handler 形成两级加载：config_handler 解析 YAML 路径映射，
# 本模块再读取映射所指向的文本正文，供 Agent 或 RAG PromptTemplate 使用。


def load_system_prompts():
    """读取普通客服场景的系统提示词。

    Returns:
        完整 Prompt 字符串，作为 ``create_agent`` 的默认 system_prompt。

    Raises:
        KeyError: prompts.yml 缺少 ``main_prompt_path``。
        OSError: Prompt 文件不存在、无法读取或编码不匹配。
    """
    try:
        # 跨文件传参：配置值 prompts_conf["main_prompt_path"] →
        # utils/path_tool.py 中 get_abs_path(relative_path) 的形参 relative_path。
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        # 配置键缺失与文件读取失败分开记录，便于快速定位是哪一层问题。
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        # 未使用 with 也能读取，但文件对象由运行时回收；正式代码通常更推荐 with。
        # read() 不做模板渲染，只原样读取整份文本；工具选择规则仍由模型解释执行。
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    """读取 RAG 内部“根据参考资料总结”的 Prompt 模板。

    返回文本稍后交给 ``PromptTemplate.from_template``，其中 ``{input}`` 与
    ``{context}`` 会分别替换为查询和召回资料。缺少变量会使输入无法被使用，
    多出未提供的模板变量则会在渲染时产生错误。
    """
    try:
        # 跨文件传参：配置值 prompts_conf["rag_summarize_prompt_path"] →
        # utils/path_tool.py 中 get_abs_path(relative_path) 的形参 relative_path。
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    """读取个人使用报告场景的系统提示词。

    该函数不会判断当前是否为报告场景；选择时机由 ``report_prompt_switch``
    中间件负责。每次调用都会重新读取文件，因此新调用可看到磁盘上的最新内容。
    """
    try:
        # 跨文件传参：配置值 prompts_conf["report_prompt_path"] →
        # utils/path_tool.py 中 get_abs_path(relative_path) 的形参 relative_path。
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e


if __name__ == '__main__':
    # 直接运行时打印报告 Prompt，验证 YAML 映射和 UTF-8 文件读取是否正常。
    print(load_report_prompts())

