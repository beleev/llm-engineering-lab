# Agent 扩展阅读：从执行到验证与改进

新内容默认归入扩展层, 不增加原主线的必读章节。先学 m13 的终态与过程评估, 再按需要进入下列实验。

| 模块 | 读完能解释什么 |
|---|---|
| [记忆更新 · 来源、范围与版本](../llm_agent/m20_memory_lifecycle/README.md) | 记忆要保留对象、范围和证据; 新事实只替换它能证明的那一部分。 |
| [完成验收 · 说完不等于做完](../llm_agent/m21_completion_verification/README.md) | 把 final 当候选结果; 只读验收通过才结束, 失败后在预算内修复。 |
| [故障恢复 · 重试、幂等与取消](../llm_agent/m22_runtime_reliability/README.md) | 超时表示不知道结果; 幂等确认副作用, 取消确认执行边界。 |
| [上下文成本 · 压缩与缓存一起算](../llm_agent/m23_context_cache/README.md) | 比较净成本, 同时检查范围、否定和来源是否保留。 |
| [检索证据 · 从相关到足够回答](../llm_agent/m24_agentic_rag/README.md) | 别名、对象、版本与答案形成完整证据链, 才能输出结果。 |
| [轨迹学习 · 先验证，再造训练数据](../llm_agent/m25_trajectory_learning/README.md) | 结果、来源、授权和预算一起决定轨迹能否进入训练集。 |
| [持续改进 · 学会、迁移、变化与保持](../llm_agent/full_learning_loop/README.md) | 先行动再读反馈, 把迁移、规则更新和旧能力保持分开验收。 |

每个模块都有对照、可失败断言和边界说明。网页用相同机制重算, `tests/test_agent_extensions.py` 核对控制项覆盖范围内的 Python / JavaScript 数字。诊断轨迹由 `python web/scripts/export-agent-evidence.py` 从真实教学 Agent 导出。

来源索引见各模块延伸阅读。书中原则按本库的标准库、固定输入、单概念实验方式重新实现, 不直接搬运依赖栈或实验收益数字。

## 验证方式

```bash
python -m llm_agent.run_all
python -m pytest tests/test_agent_extensions.py
# 安装 dev 依赖后, 全仓库快速回归 (避免可选模型模块调用真实 API)
env -u ANTHROPIC_API_KEY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
```

网页沿用 `web/package.json` 中的 source、math、quiz、evolution、build 和 bundle 检查。浏览器验收另需可选开发工具, 不加入 Agent 运行依赖:

```bash
python -m pip install playwright
python -m playwright install chromium
# 另一个终端: cd web && npm run dev
python web/scripts/check-agent-ui.py
# 要更新真实截图时, 再加 --screenshots
```

脚本检查 8 个路由上的 9 个新实验台, 覆盖 select 全选项、关键数值、滑杆与按钮的键盘操作、390px 窄屏、明暗主题和直接刷新。截图见 [章节预览](screenshots/README.md)。

训练桥接当前交付的是可验证数据、偏好对与过程标签, 并未训练权重; 费用与调度时间均明确标成模型值。

## 全书范围继续扩展

m26–m44 的逐节对照、实际执行层次与研究边界见 [知识覆盖表](agent-book-coverage.md)。其中 m32 / m37 / m44 使用仓库已有 PyTorch 依赖实际训练, 其余新增默认 demo 使用标准库。m35 的 Chromium 验收需单独安装可选 playwright 与 pillow。
