# 工程实验指南

9 组工程实验、49 个执行条件，覆盖模型交付、文档生命周期、协议、训练队列、发布、性能、调度、恢复与缓存身份。

每组实验提供原理、对照条件、自测与源码。网页展示 Python 实际执行记录；切换条件不调用模型 API，也不在浏览器训练模型。

## 实验入口

| 主题 | 运行模块 | 条件数 |
| --- | --- | --- |
| [模型交付：新进程里还是同一个模型吗](../llm_finetune/run_finetune/artifact_delivery/README.md) | `llm_finetune.run_finetune.artifact_delivery.demo` | 5 |
| [文档生命周期：旧答案什么时候必须失效](../llm_agent/m54_document_lifecycle/README.md) | `llm_agent.m54_document_lifecycle.demo` | 6 |
| [MCP：旧握手、新报文与资源端授权](../llm_agent/m55_mcp_versions/README.md) | `llm_agent.m55_mcp_versions.demo` | 8 |
| [异步 RL：样本来自哪一版策略](../llm_finetune/run_finetune/async_rollout/README.md) | `llm_finetune.run_finetune.async_rollout.demo` | 4 |
| [评测到发布：均分变好，为什么仍要回滚](../llm_agent/m56_release_loop/README.md) | `llm_agent.m56_release_loop.demo` | 6 |
| [性能实测：模型估算与 profiler 怎样对照](../llm_infer/m32_profiling/README.md) | `llm_infer.m32_profiling.demo` | 2 |
| [多租户调度：总体吞吐会掩盖谁的等待](../llm_infer/m33_tenant_scheduling/README.md) | `llm_infer.m33_tenant_scheduling.demo` | 4 |
| [任务恢复：旧检查点遇到新代码](../llm_agent/m57_checkpoint_migration/README.md) | `llm_agent.m57_checkpoint_migration.demo` | 7 |
| [缓存身份：相同 token 不一定能复用](../llm_infer/m34_cache_identity/README.md) | `llm_infer.m34_cache_identity.demo` | 7 |

## 运行与校验

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python web/scripts/export-engineering-completion.py --check
python scripts/check-engineering-coverage.py
```

实验清单见 [engineering-completion-manifest.json](engineering-completion-manifest.json)。

## 实验边界

- **模型交付**：拒绝时误差显示 0 代表未运行推理；int8 是反量化后的误差实验，没有整数算子加速。manifest 需由可信渠道提供；签名、真实 adapter 基座指纹与跨引擎容差属于部署扩展。
- **文档生命周期**：两维固定编码器与单桶 ANN 用于隔离权限、召回和失效行为，不代表真实语义效果。HTML 保留字符偏移；PDF/扫描件必须由单独解析/OCR 适配器提供页码与坐标，本模块显式拒绝原始 PDF。缓存按租户与索引代际隔离，变更时保守全量失效。
- **MCP**：HTTP 资源端真实运行于 loopback，固定时钟和 opaque 凭据用于授权反例；未实现 OAuth 登录、令牌签发、SSE 或完整互操作认证。取消/超时检验工具执行层的合作停止，不等于网络断连自动撤销副作用。stdio 的环境凭据与 HTTP 授权流程分开。
- **异步 RL**：这是异步执行顺序的确定性重放，不是多 GPU 吞吐实测。ESS 是有效样本比例；裁剪会引入偏差。拒收时期望 0 表示没有更新，不能解释成策略奖励降为零。引擎温度与数值精度差异应保留实际行为 logp，不能用训练器重算值替换。
- **评测到发布**：固定请求回放模拟 10% 灰度，没有操作真实流量。复用配对 bootstrap；平均提升不能抵消关键切片退化。回归样本按规范化文本去重并排除受保护集合，仍需实体级去重和隐私审核；延迟反馈与缺失分别计数。
- **性能实测**：CPU 分配字节是正向分配事件之和，不是峰值驻留内存。profiler 会扰动耗时，计时与采样分开；CPU self 时间排除子事件但不能直接与异步 GPU 总时间相加。CUDA 选做会同步后计时并记录峰值显存；默认测量设备为 CPU。
- **多租户调度**：离散服务量模拟，不含真实 GPU、KV 或 token 预测。首次获得服务的样本才有等待读数，需连同从未获服务的超时/拒绝一起读；有限轨迹不能证明任意负载下无饥饿。
- **任务恢复**：真实文件、SQLite 和进程强退；不恢复任意 Python 栈，不实现分布式租约或任意 schema 自动迁移。原子重命名未对目录 fsync，不承诺断电持久性。
- **缓存身份**：命中 1 / 未命中 5 是合成代价，不是测量时延或攻击成功率。salt 只隔离缓存键，不隔离服务进程、网络和显存；多模态还要绑定内容摘要。

## 维护

修改实验源码或清单后，运行导出器更新回放，再用 `--check` 核对。源码摘要覆盖教学包；执行记录中的参数、指标与模块 README 应一致。浏览器交互与页面布局的检查命令见 [验证指南](validation.md)。
