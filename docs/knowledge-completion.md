# 九项教学缺口补齐记录

2026-11-08；基于复查提交 `db741f2`，对照 ai-agent-book `dbc046eb896ac4e39aa19c7774c8bf49583b89a6`。本记录关闭的是[知识复查](knowledge-audit.md)确认的九项教学缺口，不宣称全领域知识穷尽，也不把教学实验等同生产实现。

## 逐项证据

| 项 | 现在能解释/验证的问题 | 正文和代码 | 网页入口 | 失败对照与边界 |
| --- | --- | --- | --- | --- |
| 1. Chat Template | 角色/正文/结束标记如何变成 token；多轮 labels 左移与监督位置 | [讲义](../llm_finetune/run_finetune/chat_template/README.md)、[实现](../llm_finetune/data/chat_template.py) | finetune-sft | 错误角色、工具 id/name 错配、未完成调用、重复 assistant、超长、空监督拒绝；字节模板不是厂商 tokenizer |
| 2. trace/span | task→model→attempt 与 tool 的父子树；并发、重试、取消如何计数 | [m31](../llm_agent/m31_eval_suite/README.md)、[真实调用链实验](../llm_agent/m31_eval_suite/trace_demo.py) | agent-evals / agent-eval-suite | 线程屏障证明工具重叠；取消关闭所有 span；不存秘密正文；失败尝试账单未知，未实现 OTLP |
| 3. RFT | 当前策略采样→独立执行验证→成功轨迹 CE 更新 | [讲义与训练](../llm_finetune/run_finetune/rft/README.md) | finetune-sft / agent-trajectory-learning / agent-agent-training | 同起点 SFT/RFT/RL；256 次候选仅保留 18 条、覆盖 10/16 任务；零成功不更新；协议内留出不是语义泛化 |
| 4. 真实接口运行时 | 实际适配器的流式累积、工具参数闭合、重试和服务端 usage | [m15](../llm_agent/m15_claude_api/README.md)、[流式解析器](../llm_agent/core/streaming.py) | agent-runtime-reliability / agent-async-runtime / agent-full-loop | 完整/断流/非法 JSON；限流前重试、开始后不自动重发；未知类型报错；离线协议验证，未调用在线服务 |
| 5. 语义检索与重排 | 可训练表示、对比正负样本、联合重排；表征/索引/候选误差分开 | [m45](../llm_agent/m45_semantic_retrieval/README.md) | agent-rag / agent-vector-index | 改写、跨语言、否定、编号探针；top-4 未召回答案时重排失败，扩大范围才能接触答案；有限词表，非通用语义模型 |
| 6. 多维并行预算 | DP×TP×PP rank 约束、每卡状态/激活、α/β 通信、重叠、Roofline/MFU | [m22](../llm_train/m22_parallel_budget/README.md) | train-model-parallel / train-collectives-loop | TP 更大状态减半但步时间上升；非法网格拒绝；所有时间是代价模型，未启动多 GPU |
| 7. 服务容量 | 到达率→排队→准入→尾延迟→goodput；取消/超时回收资源 | [m28](../llm_infer/m28_service_capacity/README.md) | infer-scheduler / infer-multi-replica-routing / infer-engine | 8/24/48/96 请求/s、有界/无界；终态守恒和最终零占页；完成样本分位数需与拒绝/超时并读，非硬件实测 |
| 8. 多模态入口 | patch/mel→编码器→对齐/前缀/cross-attention→文本/codec 输出 | [阅读路线](multimodal-learning-path.md)、已有四套模型讲义 | 阶段 2 总览“图像与音频怎样接入语言模型” | CLIP / Qwen2-VL / Whisper / Omni 顺序、真源码入口、四道题；保留随机批次、缺音频前处理/codec 等边界 |
| 9. 感受野 | 理论可达、可接受上下文、实际有效利用的区别 | [Mistral](../llm_models/run_models/language_models/mistral/README.md)、[方法](../llm_models/models/language_models/mistral.py) | basic-forward / models-mtp / models-gptoss / 术语表 | 统一 W 含当前位置时 `1+L*(W-1)`，位置 t 截到 t+1；4层×8窗口是29，跨层集合传播测试核对，不承诺有效利用长度 |

## 保持原有学习结构

六阶段和 26 个二级主题不变；没有新增阅读路由。七组可运行实验并入 15 个既有章节，使用原 LabFrame；同一章节可有多个补充实验。多模态阅读组件放在模型总览，复用原模型和脚本。新增 13 道自测题，题库共 359 道。

网页明确展示“Python 执行记录回放”：点击条件读取离线实际运行结果，不在浏览器重训模型或调用 API。`export-knowledge-completion.py` 从七组实验的 30 个条件生成 JSON，记录全部 Python 课程源码和导出器的摘要；`--check` 重跑并逐字比对，防止修改源码后仍显示旧数字。时间戳和随机 trace id 不放进固定记录。

开发服务器还增加了仓库 Python 目录监听，解决在 web 根目录外新建代码后，源码展开列表不自动更新的问题。不会改变发布包的加载方式。

## 验证方式

以下命令均从仓库根运行。实测结果见本节末的验收记录；测试明确移除 API key，避免自动调用付费服务。CPU 多线程库固定为 1 线程，避免共享机器上小模型的线程调度开销掩盖执行结果。

```bash
env -u ANTHROPIC_API_KEY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
env -u ANTHROPIC_API_KEY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -m slow -k train_rft
python web/scripts/export-knowledge-completion.py --check
python scripts/check-agent-book-coverage.py --book /data00/ai-agent-book
python web/scripts/check-knowledge-completion-ui.py
python web/scripts/check-navigation-ui.py --base http://127.0.0.1:5174/llm-engineering-lab/
```

web 目录运行 `npm run check:sources`、`check:math`、`check:quiz`、`check:evolution`、`build`、`check:bundle`。新增 UI 检查覆盖每个实验条件、深浅主题、1440/320 宽度、键盘选中、读数与源码；导航回归继续检查 111 个目录入口的编号、缩进、对比度、桌面/移动端交互。

## 仍保留的工程边界

线上服务接口和计费没有实际验收；缓存控制、跨请求限流、全局 deadline、完整服务端工具事件未实现。语义模型是任务内训练，RFT 没有新任务域泛化证据。并行/容量数字不是 GPU 测量，多模态没有真实数据集质量报告。原有 MCP HTTP/授权、A2A 网络、OS 沙箱等生产深化边界也仍保留。

这次补齐的是“能从概念走到机制、失败对照和可复现证据”的教学环节。287 个书中标题映射继续作为索引，不能代替逐项语义审查。

## 本轮验收记录

- 全仓快速回归：194 项通过；39 项 slow 未整体重跑，本轮相关的 RFT 训练另行验收。最后增加的非有限 JSON 边界及相关 Agent 路径由下述专项回归覆盖。

- 新增边界测试 20 项：模板监督/配对/截断、流式累计 usage/完整性/非有限 JSON、重试预算、trace 父子树/取消、空 RFT 更新、并行约束、容量守恒、感受野集合传播、检索候选/OOV。与原 Agent 扩展测试一同运行，68 项通过。
- RFT 训练单独验收：1 项通过；SFT/RFT/RL 参数真实更新，RFT 与 RL 各采样 256 次。
- 7 组、30 条件回放：重跑结果与源码摘要完全一致；默认标准库 Agent run_all 的 44 个 demo 通过。
- 网页：285 个源码引用、1473 个公式、359 道题（长度提示检查）、100 个演进链章节通过；构建通过，入口约 184.45 kB，正文和题库仍按阶段加载。
- 新页面：15 个现有路由入口、120 次条件/主题/视口组合、4 个多模态入口全部通过，源码可展开，无 JS 错误。
- 原导航：111 个目录入口，深浅主题，桌面侧栏 200/260/480，移动端 320/390/768，编号、缩进、对比度、键盘与返回/重载通过。
- 开发监听：实际新建并删除临时 Python 文件，源码 glob 在不重启服务时同步更新；临时文件已清理。
- 原书对照：287 个标题、本地证据路径与固定版本源文一致。
