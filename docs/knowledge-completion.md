# 机制实验指南

本指南覆盖多轮模板、调用追踪、可验证训练、模型接口、检索、并行、容量、多模态与感受野。每项列出讲解、代码、网页主入口和验证边界。

| 项 | 讲解与验证的问题 | 正文和代码 | 网页入口 | 失败对照与边界 |
| --- | --- | --- | --- | --- |
| 1. Chat Template | 角色/正文/结束标记如何变成 token；多轮 labels 左移与监督位置 | [讲义](../llm_finetune/run_finetune/chat_template/README.md)、[实现](../llm_finetune/data/chat_template.py) | finetune-sft | 错误角色、工具 id/name 错配、未完成调用、重复 assistant、超长、空监督拒绝；字节模板不是厂商 tokenizer |
| 2. trace/span | task→model→attempt 与 tool 的父子树；并发、重试、取消如何计数 | [m31](../llm_agent/m31_eval_suite/README.md)、[真实调用链实验](../llm_agent/m31_eval_suite/trace_demo.py) | agent-eval-suite | 线程屏障证明工具重叠；取消关闭所有 span；不存秘密正文；失败尝试账单未知，未实现 OTLP |
| 3. RFT | 当前策略采样→独立执行验证→成功轨迹 CE 更新 | [讲义与训练](../llm_finetune/run_finetune/rft/README.md) | agent-trajectory-learning | 同起点 SFT/RFT/RL；256 次候选仅保留 18 条、覆盖 10/16 任务；零成功不更新；协议内留出不是语义泛化 |
| 4. 真实接口运行时 | 实际适配器的流式累积、工具参数闭合、重试和服务端 usage | [m15](../llm_agent/m15_claude_api/README.md)、[流式解析器](../llm_agent/core/streaming.py) | agent-async-runtime | 完整/断流/非法 JSON；限流前重试、开始后不自动重发；未知类型报错；离线协议验证，未调用在线服务 |
| 5. 语义检索与重排 | 可训练表示、对比正负样本、联合重排；表征/索引/候选误差分开 | [m45](../llm_agent/m45_semantic_retrieval/README.md) | agent-vector-index | 改写、跨语言、否定、编号探针；top-4 未召回答案时重排失败，扩大范围才能接触答案；有限词表，非通用语义模型 |
| 6. 多维并行预算 | DP×TP×PP rank 约束、每卡状态/激活、α/β 通信、重叠、Roofline/MFU | [m22](../llm_train/m22_parallel_budget/README.md) | train-model-parallel | TP 更大状态减半但步时间上升；非法网格拒绝；所有时间是代价模型，未启动多 GPU |
| 7. 服务容量 | 到达率→排队→准入→尾延迟→goodput；取消/超时回收资源 | [m28](../llm_infer/m28_service_capacity/README.md) | infer-multi-replica-routing | 8/24/48/96 请求/s、有界/无界；终态守恒和最终零占页；完成样本分位数需与拒绝/超时并读，非硬件实测 |
| 8. 多模态入口 | patch/mel→编码器→对齐/前缀/cross-attention→文本/codec 输出 | [阅读路线](multimodal-learning-path.md)、四套模型讲义 | 阶段 2 总览“图像与音频怎样接入语言模型” | CLIP / Qwen2-VL / Whisper / Omni 顺序、真源码入口、四道题；保留随机批次、缺音频前处理/codec 等边界 |
| 9. 感受野 | 理论可达、可接受上下文、实际有效利用的区别 | [Mistral](../llm_models/run_models/language_models/mistral/README.md)、[方法](../llm_models/models/language_models/mistral.py) | attention / basic-forward / models-gptoss / 术语表 | 统一 W 含当前位置时 `1+L*(W-1)`，位置 t 截到 t+1；4层×8窗口是29，跨层集合传播测试核对，不承诺有效利用长度 |

## 执行记录与源码

七组回放实验包含 30 个条件。网页切换条件读取离线执行结果；`export-knowledge-completion.py` 实际运行实验并记录教学源码摘要。`--check` 重新执行并比较结果，时间戳和随机 trace id 不进入固定记录。

开发服务器监听仓库 Python 目录，源码展开列表随文件变化更新。多模态阅读组件使用阶段 2 的模型与脚本。

## 验证

```bash
env -u ANTHROPIC_API_KEY OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_knowledge_completion.py
python web/scripts/export-knowledge-completion.py --check
python web/scripts/check-knowledge-completion-ui.py
```

浏览器检查需要运行 Web 服务，完整命令见 [验证指南](validation.md)。

## 验证边界

流式接口使用离线协议夹具，未验收在线服务或真实计费。语义模型是任务内训练，RFT 没有新任务域泛化证据。并行与容量数字来自代价模型；多模态脚本验证信息通路与训练目标，不提供真实数据集质量报告。MCP、A2A 和沙箱的实现范围见各模块 README。
