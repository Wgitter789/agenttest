"""为整个工程提供统一的绝对路径。

通过当前文件自身的位置推导项目根目录，使配置、Prompt、数据和日志路径不再
依赖用户从哪个目录启动程序。注意：只有实际调用本工具转换过的路径才具备
这个特性；Chroma 的 persist_directory 当前没有使用它。
"""

import os


def get_project_root() -> str:
    """返回项目根目录的绝对路径。

    当前文件位于 ``项目根目录/utils/path_tool.py``，因此先取得所在 ``utils``
    目录，再向上一级即可得到项目根目录。
    """
    # __file__ 是本模块文件路径；abspath 将它规范为绝对路径。
    current_file = os.path.abspath(__file__)

    # 第一次 dirname：去掉 path_tool.py，得到 utils 目录。
    current_dir = os.path.dirname(current_file)

    # 第二次 dirname：去掉 utils，得到项目根目录。
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path: str) -> str:
    """把项目内相对路径转换为绝对路径。

    Args:
        relative_path: 相对项目根目录的路径，例如 ``config/rag.yml``。

    Returns:
        使用当前操作系统分隔符拼接的路径字符串。
    """
    project_root = get_project_root()
    # os.path.join 不会检查目标是否真实存在，也不会创建目录或文件。
    # 若 relative_path 本身是绝对路径，在 Windows 上它可能覆盖前面的 project_root，
    # 因而该函数约定调用者传入可信的项目相对路径。
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    # 最小演示：输出项目 config 目录下某文件应有的绝对路径。
    print(get_abs_path("config/config.txt"))
