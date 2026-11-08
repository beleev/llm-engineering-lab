# Chat Template：一段对话究竟在教哪些 token

## 直觉

API 的 role/content 是消息对象；模型只接收 token。模板负责把角色、正文、结束位置写成同一条序列。训练时和生成前缀使用不同模板，就像老师和考卷用了不同的标点规则。

## 核心原理

本实验用 UTF-8 字节加 32 编码正文，0–8 保留给控制标记，不冒充任何厂商 tokenizer。消息格式是 `角色 + HEADER_END + 正文 + END/EOS`。普通 assistant 回复以 EOS 结束；工具调用以 END 暂停，工具结果之后再继续 assistant。正文写下字符串 `EOS` 不会变成控制 token。

`serialize` 同时生成 token 与监督标志。`training_example` 左移后令 `labels[t]=ids[t+1]`，只保留 assistant 正文和结束标记。用户、system、工具结果、角色头和 padding 的 labels 都是 -100。attention mask 则只屏蔽 padding，不能把不计算 loss 的上下文也遮掉。

工具结果必须匹配待处理调用的 id 和工具名。超长输入直接拒绝，不截半段工具 JSON。实际生产可先裁剪完整历史轮次，再重新套模板。本实现每轮只支持一个工具调用。

## 运行

```bash
python -m llm_finetune.run_finetune.chat_template.demo
```

## 运行后应该看到什么

正确的七轮样本打印 token 和左移后的 labels：172 个 token、75 个监督位置、两个被监督的 EOS。训练序列与推理 assistant 前缀逐位一致。把工具结果伪装成 user 或把上限设为 40 都失败。右 padding 的 labels 为 -100；已有生成工具仍负责左 padding 与批量 EOS。

## 常见误区

- 不计算用户 loss 不代表模型看不到用户输入。attention mask 和 loss mask 的用途不同。
- 不应在模板已经生成的 EOS 后再手工追加结束标记；本接口只收结构化 Turn，不收拼好的 token。
- 工具结果是观察，不是 assistant 的监督答案；工具调用本身则属于 assistant 动作。

## 自测题

一个回复的第一字对应哪个 labels 位置？为什么不能把工具返回的十万字也当 assistant 答案？

答案：位于该字前一 token 的预测位置；否则模型会学习伪造工具观察，且工具输出量会改变训练目标的权重。
