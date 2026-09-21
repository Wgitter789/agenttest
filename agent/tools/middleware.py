"""Agent 自定义中间件。

中间件不是独立运行的服务，而是挂在 ``create_agent`` 执行图上的钩子。
本文件分别演示：

- ``wrap_tool_call``：包围每次工具执行；
- ``before_model``：在每次聊天模型调用前运行；
- ``dynamic_prompt``：根据本次运行上下文动态生成系统提示词。
"""

from typing import Callable

from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
# 三个装饰器把普通函数注册到 Agent 生命周期的不同位置：
# wrap_tool_call 包围工具调用，before_model 在模型调用前观察/修改状态，
# dynamic_prompt 在每轮模型调用前决定系统提示词。
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
def monitor_tool(
        # ToolCallRequest 封装了工具名、参数、工具对象、状态和 runtime 等信息。
        request: ToolCallRequest,
        # handler 是“继续执行真正工具”的回调。中间件必须调用它，工具才会运行。
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """记录每次工具调用，并处理报告模式的上下文开关。

    Wrap 类型中间件的结构类似：执行前逻辑 → handler(request) → 执行后逻辑。
    如果 handler 抛出异常，本函数记录错误后继续向上抛出，因此当前实现不会把
    工具错误自动转换成模型可阅读的友好消息。
    """
    # 在真正执行工具前记录名称和模型生成的参数，便于排查“选错工具/传错参数”。
    # request.tool_call 是模型生成的结构，通常含 name、args、id；这里使用前两个字段。
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        # 传参：当前形参 request → handler(request) 的形参 request。
        # handler 再根据 request.tool_call["args"] 的键名，把各值传给目标工具的同名形参；
        # 例如 {"city": "杭州"} 最终会传到 agent_tools.get_weather(city) 的 city。
        result = handler(request)
        # 此处的“成功”仅指 handler 正常返回，并未检查结果是否包含业务错误信息。
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        # fill_context_for_report 是一个信号工具。只有它成功执行后才打开报告标志，
        # 保证下一轮 dynamic_prompt 能选择报告专用 Prompt。
        if request.tool_call['name'] == "fill_context_for_report":
            # 依赖 execute_stream 传入可修改的字典 context；这里直接修改共享字典，
            # 没有向 messages 追加信息，也没有返回用于更新图状态的 Command。
            # 该 context 正是 react_agent.py 调用 self.agent.stream(..., context=...) 时
            # 传入的实参；这里修改其中 report 键，供下方 report_prompt_switch 读取。
            request.runtime.context["report"] = True

        # 原样返回 handler 的结果非常关键：ToolMessage 会进入消息历史供模型阅读；
        # Command 则还能携带状态更新或图跳转信息。
        return result
    except Exception as e:
        # 日志保留失败工具及原因，然后保持原异常语义交给上层处理。
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,          # Agent 状态，核心字段 messages 保存完整消息轨迹。
        runtime: Runtime,           # 本次执行的 context、store、stream writer 等运行信息。
):
    """在每次模型调用前记录当前消息数量与最后一条消息。

    一次用户请求可能多次进入这里：首次让模型决策，工具返回后再次让模型判断，
    直到模型不再请求工具并给出最终回答。
    """
    # INFO 同时输出到控制台和日志文件（由 logger_handler 的级别设置决定）。
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    # DEBUG 默认只写入日志文件。type(...) 可区分 HumanMessage、AIMessage、
    # ToolMessage；content 展示上一节点给模型的正文。
    # 下方表达式要求 messages 非空且 content 为字符串；即使日志级别不输出 DEBUG，
    # f-string 仍会先求值，因此调低日志级别不会跳过下标访问和 strip 调用。
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    # before_model 可以返回状态更新；这里仅观察、不修改，所以返回 None。
    return None
    # 返回 None 表示不更新 AgentState；若需修改状态，应返回框架约定的更新结构。


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """根据 runtime context 为当前模型调用选择系统提示词。

    ``dynamic_prompt`` 在每一次模型调用前都会执行，不只在一次 Agent 请求开始时
    执行。因此，工具中途改变 report 标志后，紧接着的模型调用即可看到新 Prompt。
    """
    # get 提供默认 False，避免上下文中没有 report 键时触发 KeyError。
    is_report = request.runtime.context.get("report", False)
    # get 的默认值只处理缺失的 report 键；context 本身仍必须是支持 get 的对象。
    if is_report:
        # 返回报告提示词会替换本次模型调用使用的系统提示词，而非追加在默认提示词后。
        # 消息历史与已注册工具仍由原 Agent 执行图管理。
        # 报告场景：要求根据用户记录生成指定格式的使用情况报告。
        return load_report_prompts()

    # 普通场景：使用完整客服角色、工具说明与报告触发规则。
    return load_system_prompts()
    # dynamic_prompt 返回字符串，框架会把它作为当前轮的系统消息传给聊天模型。
