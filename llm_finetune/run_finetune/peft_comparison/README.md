# PEFT：Prompt、Prefix、Adapter 与 LoRA

## 直觉

冻结基座并不意味着只能改提示词。四种方法把可训练参数放在不同位置，影响梯度路径和部署代价。

## 核心原理

Prompt Tuning 学输入 embedding，再经过冻结的注意力投影；Prefix Tuning 直接提供每层的 K/V 前缀，真实多层模型需逐层注入。本例在同一单层基座比较两条路径。

瓶颈 Adapter 把 h→降维→非线性→升维放进残差；LoRA 给权重增加低秩 ΔW=AB，可在适用的线性层合并。前缀占用上下文/缓存，Adapter 增加算子，LoRA 合并后不一定增加线性算子。

多个 LoRA 可按权重求和，但组合效果不保证可加；不同基座、秩与目标层必须核对。Prompt/Prefix 通常不能直接当 ΔW 相加。持续个性化需比较单用户、共享和组合适配器的串扰与遗忘。

## 运行

```bash
python -m llm_finetune.run_finetune.peft_comparison.demo
```

需要 Python ≥ 3.10，依赖使用仓库的 numpy / torch；不需要模型 API。网页在现有章节中提供同一实验的执行记录回放。

## 运行后应该看到什么

以下数值来自本目录 `demo.py` 的实际执行；改变种子或配置后应重新运行，而不是沿用表格。

### Soft prompt

| 读数 | 实际结果 |
| --- | --- |
| 可训练参数 | 16 |
| 初始MSE | 1.5684 |
| 训练后MSE | 1.0521 |
| 冻结基座未变 | true |

### KV prefix

| 读数 | 实际结果 |
| --- | --- |
| 可训练参数 | 32 |
| 初始MSE | 1.5694 |
| 训练后MSE | 0.7128 |
| 冻结基座未变 | true |

### Adapter

| 读数 | 实际结果 |
| --- | --- |
| 可训练参数 | 32 |
| 初始MSE | 1.5489 |
| 训练后MSE | 0.5444 |
| 冻结基座未变 | true |

### LoRA

| 读数 | 实际结果 |
| --- | --- |
| 可训练参数 | 32 |
| 初始MSE | 1.5489 |
| 训练后MSE | 0.7207 |
| 冻结基座未变 | true |

## 怎样验证

四种模式使用相同冻结基座和固定 32 条样本，报告可训练参数数与 MSE，断言基座不变。参数量没有配平，结果只验证可学习路径，不作为方法排名。

## 与真实系统的差距

固定 32 条合成样本验证参数路径与梯度。没有留出泛化结论；Prompt Tuning 与提示词写作不是一回事。

## 常见误区与自测

本例 Prefix 与 Soft Prompt 最关键的区别是什么？

<details><summary>展开解析</summary>

Soft Prompt 经基座投影形成 K/V；Prefix 直接学习层内 K/V 状态。多层实现还要明确每层的注入位置。

</details>

## 原始资料

- [Prefix-Tuning](https://aclanthology.org/2021.acl-long.353/)
- [Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/)
