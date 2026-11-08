# 内容整理与工程流程补全

日期：2026-11-08。对应 [内容审查](tutorial-overlap-gap-audit.md) 的基线 `04fb92f`。逐项状态见 [闭环清单](engineering-completion-closure.json)。

这次修订处理审查中的重复入口、五类章节归属和 G1–G6 六项工程教学缺口。保留六阶段、26 个二级主题；知识更新和可验证推理各增加一个三级专题。网页现在共有 113 个章节入口、175 个实验入口、393 道题。

## 阅读结构

- RFT 主入口：`agent-trajectory-learning`。
- 流式工具执行主入口：`agent-async-runtime`。
- trace 主入口：`agent-eval-suite`。
- 语义检索主入口：`agent-vector-index`。
- DP×TP×PP 预算主入口：`train-model-parallel`。
- 服务容量主入口：`infer-multi-replica-routing`。

上述六组由 15 次完整展示改为 6 次，其他位置保留说明与关联链接。温度实验保留在基础采样章，掩码实验保留在 Attention 页；后续章节链接到基础内容。通信主循环页改讲 collective 的布局与调用语义，不再复制预算段落。KV cache、MoE、量化、记忆和评测的跨阶段递进仍然保留。

知识编辑和参数遗忘的实验、原理、自测一起归入 `finetune-knowledge-update`；应用记忆页仍负责同意、过期、删除。工具训练数据归入 `agent-agent-training`，工具权限页保留运行时控制。形式证明与科学检验归入 `agent-verifiable-reasoning`，并链接独立验收的前置知识。护栏页明确分开执行授权、模型安全、群体切片与发布治理的证据。

## 六项补全与实际验收

| 审查项 | 实现 | 已验证的行为 |
| --- | --- | --- |
| G1 模型交付 | [artifact_delivery](../llm_finetune/run_finetune/artifact_delivery/README.md) | 真正训练小 LLaMA；独立进程恢复后 FP32 logits 差为 0；int8 反量化差约 0.008654，greedy 一致；词表、模板、基座错配被拒绝 |
| G2 文档生命周期 | [m54](../llm_agent/m54_document_lifecycle/README.md) | 原始 HTML/表格保留偏移与父块；前/后过滤分别返回 1/0 个合法候选；更新、撤权、删除后旧引用失效；编码器版本不兼容被拒绝 |
| G3 MCP 版本 | [m55](../llm_agent/m55_mcp_versions/README.md) 与 [m09](../llm_agent/m09_mcp/README.md) | 旧 stdio 子进程与新 HTTP 资源端均实际运行；未知版本报错；错误受众/过期返回 401、权限不足返回 403；执行层取消和截止时间阻止提交 |
| G4 异步 RL | [async_rollout](../llm_finetune/run_finetune/async_rollout/README.md) | 旧采样期望 0.2，经重要性校正为目标 0.8；裁剪后为 0.24，展示偏差；过期拒收、背压、取消及并发权重切换保持样本归属 |
| G5 发布回滚 | [m56](../llm_agent/m56_release_loop/README.md) | 均分提高而关键切片变差时离线拒绝；质量、延迟、成本、安全回归分别触发回滚；缺失/待到达反馈不当成成功；回流样本排除受保护集并去重 |
| G6 性能测量 | [m32](../llm_infer/m32_profiling/README.md) | 真正调用 torch.profiler；两种实现矩阵乘各 1 次、逻辑 FLOPs 相同；分配事件和算子数不同；独立计时记录环境、warmup、重复次数及 min/median/max |

六组共有 31 个条件，网页回放由实际 Python 执行生成。数据与教学说明维护在 [工程清单](engineering-completion-manifest.json)。[本机 profiler 样本](engineering-profiling-sample.json) 保留此次实际环境和读数；它是观测记录，不能用某次耗时给所有机器排序。

## 教学完成的边界

- G1 用可信 manifest 校验文件与语义一致性；哈希本身不是来源认证。示例不加载真实第三方 adapter，真实部署还需签名、基座权重指纹与跨引擎容差策略。
- G2 用 HTML 表格和单桶近似检索隔离生命周期问题；原始 PDF 被显式拒绝。PDF/OCR 适配器的页码、版面坐标和置信度要求已讲清，但没有把这个小解析器包装成通用文档服务。
- G3 是两种固定版本的报文教学与本机 HTTP 资源端反例。HTTP OAuth 登录、签发、SSE、完整规范互操作认证不在实现内。执行器取消与超时检查不证明网络断连能够撤销已经提交的副作用。
- G4 是异步事件顺序的确定性重放，加并发队列边界测试；不是多 GPU 训练或吞吐测试。重要性校正依赖行为概率正确及支持集覆盖，裁剪不能保证无偏。
- G5 用固定流量和版本模拟发布，不操作真实服务。数据回流还需实体级去重、隐私与人工标签审核；小样本区间不代表真实业务统计功效。
- G6 默认真实 CPU 实测，CUDA 分支可显式运行；本次没有 GPU 结果。CPU 正向分配总量不是驻留峰值，ATen 调用数不是 GPU kernel 数。

原审查“已讲过但应标明深度”的多模态、具身、形式验证、编辑与遗忘主题，继续保留其真实模型或硬件边界；不是用新增合成实验替代这些边界。GRPO 已声明的 DAPO overlong reward shaping 等论文实现差异也继续显式保留。

## 复现与维护

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python web/scripts/export-engineering-completion.py --check
python scripts/check-engineering-coverage.py
python scripts/check-research-coverage.py
pytest tests/test_engineering_completion.py
```

工程回放、研究回放、旧知识回放的源码摘要覆盖教学包；修改 Python 后重新导出三份数据。重导出会实际运行实验，不是手填读数。

```bash
python web/scripts/export-engineering-completion.py
python web/scripts/export-research-completion.py
python web/scripts/export-knowledge-completion.py
```

网页沿用同一个 LabFrame、条件按钮、源码展开与自测风格。研究与工程回放共享 ExperimentReplay 组件，避免继续复制展示逻辑。构建仍将各阶段正文、题库和实验按需加载。

最终检查结果见 [验证记录](engineering-completion-validation.md)。
