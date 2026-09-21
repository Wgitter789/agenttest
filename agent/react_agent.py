"""扫地机器人客服 Agent 的组装与执行入口。

``ReactAgent`` 把聊天模型、系统提示词、工具和中间件组合成一个可运行的
LangChain Agent。模型负责决定“直接回答”还是“调用哪个工具”；LangChain
负责实际执行工具，并把工具结果继续交给模型，直到得到最终答复。
"""

# create_agent 是 LangChain 的高层组装函数：把模型、工具和中间件编译成一个
# 可执行状态图，并自动维护 HumanMessage、AIMessage、ToolMessage 的往返过程。
from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    """封装 LangChain Agent，向界面提供简单的流式执行接口。"""

    def __init__(self):
        """创建 Agent 执行图。

        ``create_agent`` 的四个核心输入分别是：

        - model：负责理解、决策、工具调用和最终回答的聊天模型；
        - system_prompt：默认客服行为规则；
        - tools：模型可以选择的外部能力；
        - middleware：在模型/工具调用前后插入的监控与控制逻辑。

        返回对象底层是由 LangGraph 执行的图，但调用方可以直接使用
        ``invoke``、``stream`` 等 LangChain 高层接口。
        """
        self.agent = create_agent(
            # 实参 chat_model → LangChain create_agent(...) 的形参 model。
            model=chat_model,
            # load_system_prompts() 的返回值 → create_agent(...) 的形参 system_prompt。
            system_prompt=load_system_prompts(),
            # 工具对象列表 → create_agent(...) 的形参 tools；模型以后按各工具形参名生成实参。
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            # tools 的排列不规定执行顺序；具体工具及参数由模型根据问题和提示词选择。
            # 三个中间件对象 → create_agent(...) 的形参 middleware。
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        # self.agent 是编译后的执行对象，不是聊天模型本身。它负责循环执行：
        # 模型判断 → 必要时调用工具 → 把工具结果交回模型 → 生成最终回答。

    def execute_stream(self, query: str):
        """执行一次独立的用户请求，并逐步产出可显示文本。

        Args:
            query: 用户本轮输入的自然语言问题。

        Yields:
            str: Agent 当前步骤最后一条消息的非空正文，末尾附加换行。

        注意：这里只构造了一个 HumanMessage，没有接收上一次请求的消息列表，
        所以当前实现没有真正的跨请求对话记忆。
        """
        # create_agent 接受兼容 OpenAI 风格的消息字典，并在内部转换成
        # LangChain 的 HumanMessage、AIMessage、ToolMessage 等消息对象。
        # messages 是 AgentState 的核心字段。每条消息必须包含角色与正文；
        # role="user" 表示这是人类输入，而非系统规则或工具返回。
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }
        # 每次执行都会重新创建下面的 context 字典；报告标志只在本次请求内生效，
        # 即使 Streamlit 复用同一个 ReactAgent 对象，下一次请求也从 False 开始。

        # 传给 LangChain 执行图的对应关系：input_dict → stream(input, ...) 的 input；
        # "values" → stream_mode；{"report": False} → context。
        # stream_mode="values" 表示每当执行图状态更新时，返回当前完整状态值。
        # context 是本次运行独有的运行时上下文，不等同于 messages 对话状态。
        # report 初始为 False；fill_context_for_report 工具执行成功后，中间件会
        # 把它改为 True，使下一次模型调用切换到报告专用系统提示词。
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            # values 模式返回每次状态更新后的完整状态快照，所以多个 chunk 的 messages
            # 通常逐渐增长；它与只返回新增事件的 updates 模式含义不同。
            # 每个 chunk 都包含截至当前步骤的消息列表；最后一条代表最新事件。
            # 当模型只发起工具调用时，AIMessage.content 可能为空，工具请求实际
            # 位于 message.tool_calls 中，因此空 content 不代表 Agent 没有工作。
            latest_message = chunk["messages"][-1]
            # 此处没有按 HumanMessage / AIMessage / ToolMessage 过滤：如果状态快照
            # 的最后一条是用户输入或工具结果，它的正文也可能被传给界面。
            # 当前实现按字符串处理 content；若模型返回结构化内容块列表，strip 会失败。
            if latest_message.content:
                # 先判断再 strip，所以只含空白的正文仍会产出一个换行符。
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    # __name__ 只有在直接执行该文件/模块时才等于 '__main__'；被 app.py import 时
    # 下面的测试代码不会执行，可避免服务启动时自动发出测试问题。
    # 独立运行本模块时，不启动 Streamlit，直接在终端测试完整报告流程。
    # 推荐从项目根目录执行：python -m agent.react_agent
    agent = ReactAgent()

    # 实参字符串 → 本文件 ReactAgent.execute_stream(self, query) 的形参 query。
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
