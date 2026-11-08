# 模型交付：新进程里还是同一个模型吗

## 直觉

权重相同而 tokenizer 或模板不同，读到的输入就不同。交付要保存模型解释输入和生成输出所需的全部约定。

## 核心原理

交付包包含架构、权重、词表与特殊 token、模板、生成参数、基座标识和量化格式。恢复训练还需要优化器与 RNG；推理交付则必须证明新进程解释输入的方式一致。

本例训练同一个小 LLaMA，再在新 Python 进程加载。FP32 比 token、logits 与 greedy 输出；int8 单独记录反量化误差，不把它说成整数算子加速。

摘要只防损坏，不能认证来源。实验在改词表后重新计算文件摘要，仍由可信 manifest 中的语义约定拒绝。真实 adapter 还需绑定基座权重摘要；跨引擎的误差门槛要用校准输入验证。

## 运行

```bash
python -m llm_finetune.run_finetune.artifact_delivery.demo
```

需要 Python ≥ 3.10；使用现有 torch 依赖。默认 CPU、本机夹具，无模型 API 调用。

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

用已有小模型导出版本化 manifest，记录文件哈希、基座和模板标识。；新进程加载，固定输入逐层核对 token IDs、logits 容差、greedy 输出；量化产物单独定义误差容限。；故意替换 tokenizer、模板或 adapter 基座，必须明确拒绝或指出不匹配，不能静默给出不同答案。

边界测试见 [test_engineering_completion.py](../../../tests/test_engineering_completion.py)。

## 常见误区与自测

同一权重换了词表顺序，文件摘要也更新了，能直接上线吗？

<details><summary>展开解析</summary>

摘要只防损坏，不能认证来源。实验在改词表后重新计算文件摘要，仍由可信 manifest 中的语义约定拒绝。真实 adapter 还需绑定基座权重摘要；跨引擎的误差门槛要用校准输入验证。

</details>

## 参考资料

- [Transformers Models：模型配置与保存加载](https://huggingface.co/docs/transformers/main_classes/model)
- [Transformers GenerationConfig：生成行为也需要配置](https://huggingface.co/docs/transformers/main_classes/text_generation)
