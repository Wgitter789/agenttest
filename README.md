# AI RAG Agent

一个基于 LangChain Agent、RAG、Chroma 和 Streamlit 的扫地机器人智能客服样例。

## 主要功能

- 基于本地 TXT/PDF 知识库的 RAG 问答
- LangChain Agent 工具调用
- 基于模拟业务数据的使用报告生成
- Streamlit 聊天页面和流式输出
- Chroma 本地向量库
- 通义千问聊天模型与 DashScope Embedding

## 快速开始

```powershell
python -m pip install -r requirements.txt
$env:DASHSCOPE_API_KEY = "你的 API Key"
python -m rag.vector_store
python -m streamlit run app.py
```

`python -m rag.vector_store` 会将 `data/` 中的知识文件切分、向量化并写入本地 Chroma。

## 项目结构

- `app.py`：Streamlit 页面入口
- `agent/`：Agent 组装、工具与中间件
- `rag/`：知识入库、检索与 RAG 总结
- `model/`：聊天模型和 Embedding 模型工厂
- `config/`：模型、向量库和 Agent 配置
- `prompts/`：客服、RAG 总结和报告提示词
- `data/`：知识库源文件与模拟业务数据

## 注意事项

- 请始终从项目根目录启动，避免产生多份 `chroma_db`。
- 不要将 `DASHSCOPE_API_KEY` 写入代码或提交到 Git。
- 当前用户 ID、位置和月份工具为教学用模拟数据。

更详细的架构、运行和调试说明见 [`项目学习与调试指南.md`](./项目学习与调试指南.md)。
