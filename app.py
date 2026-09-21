"""Streamlit Web 应用入口。

这个文件只负责界面层工作：
1. 创建聊天页面；
2. 保存当前浏览器会话中的显示记录；
3. 把用户最新问题交给 ``ReactAgent``；
4. 将 Agent 返回的文本以“打字机”效果展示出来。

注意：当前页面虽然保存并展示历史消息，但调用 Agent 时只传入最新问题，
因此这些历史消息目前属于“界面历史”，并不构成大模型真正的多轮记忆。
"""

# Streamlit 采用“脚本即页面”的运行方式：组件调用按代码顺序生成界面，用户交互后
# 整个脚本从头执行。session_state 用于保存需要跨重跑存在的数据。
import streamlit as st
# ReactAgent 是业务层门面，页面不直接接触模型、向量库或具体工具。
from agent.react_agent import ReactAgent

# Streamlit 脚本会在每次页面交互后从头执行，所以这里的页面元素也会重新创建。
st.title("智扫通机器人智能客服")  # 页面一级标题。
st.divider()  # 插入视觉分隔线，不影响业务状态。

# session_state 能跨越 Streamlit 的脚本重跑保存对象。
# 一个浏览器会话只创建一次 Agent，避免用户每发一条消息就重新初始化模型和向量库。
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# message 列表只保存用于页面回显的消息，元素格式为：
# {"role": "user" | "assistant", "content": "消息正文"}
if "message" not in st.session_state:
    st.session_state["message"] = []

# Streamlit 重跑后，按原顺序重新绘制已经显示过的聊天内容。
for message in st.session_state["message"]:
    # chat_message(role) 创建带对应头像/样式的容器，write 把正文写入该容器。
    st.chat_message(message["role"]).write(message["content"])


# 创建页面底部的聊天输入框；用户尚未提交时，prompt 的值为 None。
prompt = st.chat_input()
# chat_input 在提交后返回字符串，同时触发一次脚本重跑；未提交时返回 None。

if prompt:
    # 这里只检查字符串是否非空；仅含空格的输入也会进入该分支并提交给 Agent。
    # 先立即显示并保存用户消息，让界面反馈更及时。
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # spinner 是耗时操作期间的界面提示上下文；离开 with 后自动消失。
    with st.spinner("智能客服思考中..."):
        # 跨文件传参：这里的实参 prompt → agent/react_agent.py 中
        # ReactAgent.execute_stream(self, query) 的形参 query。
        res_stream = st.session_state["agent"].execute_stream(prompt)
        # execute_stream 是生成器：这一行只创建迭代器，后续 write_stream 消费时
        # 才真正开始模型与工具调用。因此执行过程中的异常也会在消费阶段向外传播。

        # write_stream 已经自带打字机效果，消费完生成器后会返回完整响应。
        # 不再逐字符拆分，避免高频 DOM 更新。
        response_text = st.chat_message("assistant").write_stream(res_stream)

    # 纯文本流返回 str；如果上游产生非文本对象，write_stream 会返回 list。
    if isinstance(response_text, list):
        response_text = "".join(str(item) for item in response_text)

    if response_text:
        st.session_state["message"].append({"role": "assistant", "content": response_text})
    else:
        fallback_message = "未收到有效回复，请重试。"
        st.chat_message("assistant").write(fallback_message)
        st.session_state["message"].append({"role": "assistant", "content": fallback_message})

    # 不在流结束后立即强制 rerun。下一次用户交互会自然重跑页面，
    # 从而避免前端还在收尾流式节点时就重建 React DOM。
