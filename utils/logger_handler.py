"""项目日志配置。

同一条日志可按不同级别分别输出到终端和文件：终端默认显示 INFO 及以上，
日志文件记录 DEBUG 及以上。工具调用、模型调用和入库异常都复用这里的 logger。
"""

import logging
from utils.path_tool import get_abs_path
import os
from datetime import datetime

# 跨文件传参：实参 "logs" → utils/path_tool.py 中
# get_abs_path(relative_path) 的形参 relative_path；返回的绝对路径保存到 LOG_ROOT。
LOG_ROOT = get_abs_path("logs")

# exist_ok=True 表示目录已经存在时不报错。
os.makedirs(LOG_ROOT, exist_ok=True)

# 每条日志包含时间、logger 名、级别、源文件与行号，便于从错误跳回代码位置。
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
# Formatter 只定义日志长什么样；真正决定输出位置的是下方的 Handler。


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None,
) -> logging.Logger:
    """创建或复用一个同时输出到控制台和文件的 Logger。

    Args:
        name: Logger 名称，也是默认日志文件名的前缀。
        console_level: 控制台 Handler 的最低日志级别。
        file_level: 文件 Handler 的最低日志级别。
        log_file: 自定义日志路径；为空时按当前日期自动生成。

    Returns:
        配置完成的 ``logging.Logger`` 实例。
    """
    logger = logging.getLogger(name)
    # logging 按名称缓存 Logger：同一进程里重复传入 "agent" 会取得同一个对象。

    # Logger 自身设为 DEBUG，之后再由各 Handler 决定各自实际输出的最低级别。
    logger.setLevel(logging.DEBUG)

    # Python 模块可能被多次 import，Streamlit 也会反复重跑脚本。
    # 已有 Handler 时直接返回，避免同一条日志重复显示/写入多次。
    if logger.handlers:
        # 这里检测所有已有 Handler，而不区分是否由本函数添加。若其他代码已配置
        # 同名 Logger，本函数会直接沿用，传入的新级别和日志路径也不会生效。
        return logger

    # StreamHandler 默认写到标准错误流，适合开发时即时观察 INFO/WARNING/ERROR。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)
    # 一个 Logger 可以绑定多个 Handler；发出一条记录时，各 Handler 独立判断级别。

    # 未传路径时，每天生成一个 agent_YYYYMMDD.log 文件。
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    # 文件记录更详细的 DEBUG 信息；UTF-8 确保中文日志正确保存。
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    # FileHandler 默认以追加模式打开文件，所以同一天多次启动不会覆盖旧日志。
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)
    # 当前没有显式设置 logger.propagate=False。若根 Logger 也配置了 Handler，
    # 记录还可能继续向上传播并重复输出，这取决于宿主程序的日志配置。

    return logger


# 模块级共享 logger。其他文件 ``from utils.logger_handler import logger`` 即可使用。
logger = get_logger()


if __name__ == '__main__':
    # 直接运行时验证各级别日志的显示和写入效果。
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
