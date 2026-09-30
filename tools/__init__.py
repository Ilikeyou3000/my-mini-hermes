"""工具实现层(第 4 章起陆续填充)。

约定:每个模块只写工具函数本身,注册动作统一由 tool_registry.registry
完成 —— 这样"能力"和"暴露给模型的 schema"分开演进。

计划中的模块:terminal.py(shell 执行)、file_tools.py(读写文件)、
memory_tool.py(保存/检索记忆)、session_search_tool.py(跨会话召回 ch9)。
"""