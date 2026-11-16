# 模型交付: 新进程里还是同一个模型吗

## 直觉

同一份权重在新进程里重新加载: FP32 的 token、greedy 输出全一致, logits 最大误差 0.0; int8 反量化后误差 0.008654。把词表里 a、b 的顺序对调, 文件摘要也跟着更新, 加载仍被拒绝。权重只是数字, 交付包要把解释输入、生成输出的全部约定一起带上。

## 核心原理

交付包里要有架构配置、权重、词表与特殊 token、模板、生成参数、基座标识和量化格式。恢复训练还要加上优化器状态与 RNG; 推理交付则要证明新进程解释输入的方式和训练时一致。

本例训练一个小 LLaMA 25 步, 再在新的 Python 进程里加载。FP32 比 token、logits 与 greedy 输出三项; int8 单独记录反量化误差 0.008654, 不把它说成整数算子加速。

文件摘要只防损坏, 不能认证来源: 实验改词表后重新计算摘要, 仍被可信 manifest 里的约定副本拒绝。真实 adapter 还要绑定基座权重摘要; 跨引擎的误差门槛要用校准输入验证。

## 运行

```bash
python -m llm_finetune.run_finetune.artifact_delivery.demo
```

需要 Python ≥ 3.10; 使用现有 torch 依赖。默认 CPU、本机夹具, 无模型 API 调用。

## 运行后应该看到什么

### 完整 FP32

| 读数 | 实际结果 |
| --- | --- |
| 加载拒绝 | false |
| token 一致 | true |
| greedy 一致 | true |
| logits 最大误差 | 0.0 |


### int8 反量化

| 读数 | 实际结果 |
| --- | --- |
| 加载拒绝 | false |
| token 一致 | true |
| greedy 一致 | true |
| logits 最大误差 | 0.008654 |


### 词表错配

| 读数 | 实际结果 |
| --- | --- |
| 加载拒绝 | true |
| token 一致 | false |
| greedy 一致 | false |
| logits 最大误差 | 0.0 |


### 模板错配

| 读数 | 实际结果 |
| --- | --- |
| 加载拒绝 | true |
| token 一致 | false |
| greedy 一致 | false |
| logits 最大误差 | 0.0 |


### 基座错配

| 读数 | 实际结果 |
| --- | --- |
| 加载拒绝 | true |
| token 一致 | false |
| greedy 一致 | false |
| logits 最大误差 | 0.0 |


## 与真实系统的差距

拒绝时误差显示 0 代表未运行推理；int8 是反量化后的误差实验，没有整数算子加速。manifest 需由可信渠道提供；签名、真实 adapter 基座指纹与跨引擎容差属于部署扩展。

## 怎样验证

固定同一份小模型和同一条输入, 比较 FP32、int8 和三种故意错配 (词表、模板、基座) 在新进程里的加载结果; 前两种要求 token、greedy 一致且 logits 误差在容差内, 后三种必须明确拒绝, 不能静默给出不同答案。

边界测试见 [test_engineering_completion.py](../../../tests/test_engineering_completion.py)。

## 常见误区与自测

同一份权重换了词表顺序, 文件摘要也跟着更新了, 能直接上线吗?

<details><summary>展开解析</summary>

不能。摘要只能证明文件没损坏, 证明不了约定没变: 实验里改完词表重算摘要, 加载仍被 manifest 里的约定副本拒绝。真实交付还要把 adapter 绑到基座权重摘要, 跨引擎的误差门槛用校准输入验证。

</details>

## 参考资料

- [Transformers Models：模型配置与保存加载](https://huggingface.co/docs/transformers/main_classes/model)
- [Transformers GenerationConfig：生成行为也需要配置](https://huggingface.co/docs/transformers/main_classes/text_generation)
