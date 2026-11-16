# PEFT: Prompt、Prefix、Adapter 与 LoRA

## 直觉

同一个冻结的单层注意力基座、同样 32 条样本, soft prompt 只有 16 个可训练参数, MSE 从 1.5684 降到 1.0521; prefix、Adapter、LoRA 各 32 个参数, 分别降到 0.7128、0.5444、0.7207。冻结基座不等于只能改提示词: 四种方法把可训练参数放在不同位置, 梯度路径和部署代价也不同。

## 核心原理

Prompt Tuning 学 2 个输入 embedding, 它们还要经过冻结的 Q/K/V 投影; Prefix Tuning 直接提供这一层的 K/V 前缀, 真实多层模型要逐层注入。本例在同一个单层基座上比较这两条路径, 只读最后一个位置的输出。

瓶颈 Adapter 把 h→降维→非线性→升维放进输出残差; LoRA 给 V 投影加低秩分支 ΔW=AB, 在适用的线性层可以合并回权重。部署代价不同: 前缀占上下文/缓存, Adapter 多一个算子, LoRA 合并后不一定多出线性算子。

多个 LoRA 可以按权重求和, 但组合效果不保证可加; 基座、秩与目标层都要核对一致。Prompt/Prefix 通常不能直接当 ΔW 相加。持续个性化要比较单用户、共享和组合适配器的串扰与遗忘。

## 运行

```bash
python -m llm_finetune.run_finetune.peft_comparison.demo
```

需要 Python ≥ 3.10, 依赖使用仓库的 numpy / torch; 不需要模型 API。网页在现有章节中提供同一实验的执行记录回放。

## 运行后应该看到什么

以下数值来自本目录 `demo.py` 的实际执行; 改变种子或配置后应重新运行, 而不是沿用表格。

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

四种模式用同一个冻结基座、同样 32 条样本和 160 步, 报告可训练参数数、训练前后 MSE, 并断言基座权重没变; 参数量没有配平 (16 对 32), 结果只验证每条路径学得动, 不作为方法排名。

## 与真实系统的差距

固定 32 条合成样本验证参数路径与梯度。没有留出泛化结论；Prompt Tuning 与提示词写作不是一回事。

## 常见误区与自测

本例 Prefix 与 Soft Prompt 最关键的区别是什么?

<details><summary>展开解析</summary>

Soft Prompt 学的是输入向量, 要经过冻结的 Q/K/V 投影才变成 K/V; Prefix 直接学这一层里的 K/V 状态。多层实现还要说明每层的注入位置。

</details>

## 原始资料

- [Prefix-Tuning](https://aclanthology.org/2021.acl-long.353/)
- [Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/)
