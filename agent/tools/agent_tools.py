"""提供给 LangChain Agent 使用的工具集合。

带有 ``@tool`` 装饰器的函数会被转换为 LangChain Tool 对象。Agent 看到的
主要是工具名、description、参数名和类型，而不是完整 Python 实现。因此，
工具说明是否准确会直接影响模型的选择与传参。

本文件包含两类工具：

1. 真正访问项目数据的工具：RAG 检索、CSV 月度记录查询；
2. 教学用 Mock 工具：天气、位置、用户 ID、当前月份均返回固定或随机数据。
"""

import os
from utils.logger_handler import logger
# @tool 会读取函数名、类型标注和 description，生成模型可见的工具参数结构；
# 调用时 LangChain 再负责校验模型参数并执行原 Python 函数。
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

# 模块被 import 时立即创建 RAG 服务，而不是等到首次调用 rag_summarize。
# 这会连带初始化聊天模型、Embedding 模型、Chroma 和 Retriever。
rag = RagSummarizeService()
# 该对象由所有 rag_summarize 工具调用复用，避免每次查询都重新连接向量库和组装 Chain。

# 以下数组为教学用模拟数据池。随机返回意味着同一浏览器用户多次请求，
# 也可能得到不同 ID、位置或月份，不能把它们当作真实身份/系统时间。
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

# CSV 内容的进程内缓存。首次查询时加载；同一 Python 进程中的后续查询复用。
# Streamlit 进程重启后，这个字典也会重新变为空字典。
external_data = {}
# 此缓存由同一进程的调用共享，既不按浏览器会话隔离，也没有刷新时间或加锁逻辑。
# 修改 CSV 后，已有非空缓存不会自动重新加载。


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """检索扫地机器人知识库，并返回基于资料生成的总结。

    Args:
        query: 用于语义检索的核心问题或检索词。

    Returns:
        RAG 服务生成的中文纯文本答案。
    """
    # 跨文件传参：工具形参 query → rag/rag_service.py 中
    # RagSummarizeService.rag_summarize(self, query) 的同名形参 query。
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """返回指定城市的模拟天气。

    这里没有访问真实天气 API。无论传入哪个城市，天气指标都相同，只有城市名
    会变化；用途是演示“带字符串参数的工具”。
    """
    # city 原样插入结果；这里没有校验城市名称，也没有发起网络请求。
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """从三个候选城市中随机返回一个，模拟定位系统。"""
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    """从样例用户 ID 中随机返回一个，模拟登录态身份服务。"""
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """随机返回 2025 年某个月，模拟系统月份服务。

    函数名虽为“当前月份”，但并未调用 ``datetime.now()``，因此结果不代表
    真实当前日期。
    """
    return random.choice(month_arr)


def generate_external_data():
    """首次使用时，把 CSV 月度记录解析到全局 ``external_data`` 缓存。

    缓存结构：

    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }

    Returns:
        None。结果通过修改模块级 ``external_data`` 字典保存。

    Raises:
        FileNotFoundError: 配置的 CSV 文件不存在。

    注意：当前实现用 ``split(",")`` 手工拆分 CSV，只适用于当前简单样例。
    标准 CSV 字段可能包含逗号、换行和转义引号，正式项目应使用 ``csv`` 模块。
    """
    # 只有缓存为空时才读取文件；已有数据时函数什么也不做。
    # Python 中空字典为 False，已有任意键的字典为 True；这就是懒加载开关。
    if not external_data:
        # 跨文件传参：agent_conf["external_data_path"] 的配置值 →
        # utils/path_tool.py 中 get_abs_path(relative_path) 的形参 relative_path。
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            # with 块结束后会自动关闭文件，即使解析期间出现异常也一样。
            # 第一行是 CSV 表头，所以从索引 1 开始处理数据行。
            for line in f.readlines()[1:]:
                # readlines 会一次读入整个文件；每个数据行必须至少包含六列。
                # 空行或缺列会在后续 arr 下标访问时报错，当前函数不跳过这些坏行。
                arr: list[str] = line.strip().split(",")

                # 当前 CSV 每个字段外层有双引号，这里手工去掉全部双引号。
                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")
                # 所有值均保留为字符串；time 在这里表示 YYYY-MM 月份键，未解析为日期。

                # 第一次遇到某个用户时，先为它创建按月份索引的子字典。
                if user_id not in external_data:
                    external_data[user_id] = {}

                # 相同用户和月份再次出现时，后读到的记录会覆盖先前记录。
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }
                # 记录逐行写入全局缓存：若后续行读取失败，已写入的数据仍然保留。
                # 因为入口以“缓存非空”判断加载完成，下一次调用可能复用这个部分缓存。


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """查询指定用户、指定月份的模拟设备使用记录。

    Args:
        user_id: 数字字符串形式的用户 ID，例如 ``"1001"``。
        month: ``YYYY-MM`` 格式的月份，例如 ``"2025-01"``。

    Returns:
        找到时，当前实现实际返回包含“特征、效率、耗材、对比”的字典；找不到
        时返回空字符串。这里的 ``-> str`` 与成功分支实际类型并不一致，是示例
        项目中需要后续修正的类型问题。
    """
    # 确保 CSV 已加载。缓存已有内容时，此调用不会再次读取磁盘。
    generate_external_data()
    # 加载发生在下方 try 之外，文件缺失、编码错误、数据缺列等异常会继续向上传递。

    try:
        # 直接返回缓存中的字典对象，没有复制或格式化成字符串；键采用精确匹配，
        # 例如月份 "2025-1" 不会匹配 CSV 中的 "2025-01"。
        return external_data[user_id][month]
    except KeyError:
        # 用户不存在和月份不存在都会进入这里；工具不抛错，让 Agent 可以处理空结果。
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    """发出“进入报告模式”的信号。

    函数本身只返回一段确认文字。真正的状态修改发生在 ``monitor_tool``
    中间件中：它识别到本工具名后，把 runtime context 的 ``report`` 设为 True。
    这演示了工具调用如何与 Agent 的运行时上下文联动。
    """
    # 装饰器的 description 仍写着“无返回值”，但实际会返回确认字符串供工具消息使用。
    # 单独执行此工具不会切换模式；需要通过已注册 monitor_tool 的 Agent 执行。
    return "fill_context_for_report已调用"
