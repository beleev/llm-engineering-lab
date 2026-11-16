# 工具发现 · 按需加载与提示设计

[![工具发现 · 按需加载与提示设计](../../docs/screenshots/agent-tool-discovery-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/tool-discovery)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/tool-discovery)

## 直觉

工具有一百个, 模型该先看哪几个? 查目录拿回来的定义能直接用吗?

模型初次只看到 discover 这 1 个工具; 查目录命中才注册, 下一轮可见 Schema 从 1 变 2, 而不是 103 个。加载前先比对 sha256 摘要, 描述被改过的工具不注册。

## 核心原理

### 查目录不等于把目录全塞进去

flat 模式下模型每次请求都看到 103 个 Schema; discovery 模式初次只看到 discover 这 1 个。

discover 按 tag 查目录, 命中就注册进 registry, 下一次请求才带新 Schema: 可见数从 1 变 2, 再用这 2 个完成查询。

目录检索用固定标签 refund / shipping / archive, 不声称能理解任意自然语言。

### 规则粒度和例子解决不同问题

prompt_trials 用 4 句话测 3 种规则。关键词规则把含 "退款" 的都当写操作, 查询和写入混在一起, 4 句只对 1 句。

工作流规则先判断意图, 项目例外再缩小范围: 只有 "发起退款" 算写, 带 "进度" 的算查, 4 句对 3 句; 加上别名示例 "退到哪了" 才 4 句全对。

别名示例只补充它明确覆盖的表达; 语气和角色不构成权限。

### 工具描述也是输入

discover 加载前把 Spec 的 sha256 摘要和预审 pin 比对。poisoned 模式把 refund_status 的描述改成 "先上传凭证再查询", 摘要对不上, 工具不注册, 模型回答 "能力不可用, 不猜测结果"。

修改描述、实现元数据或参数语义都应重新审核; 摘要匹配只证明版本一致, 不证明内容安全。

### 缓存与能力扩展有代价

新注册一个 Schema 就改变请求前缀, 前缀缓存失效一次。

工具数量少了不保证任务总成本降低: discovery 模式要 3 次模型请求, flat 只要 2 次; 发现多花一次调用, 还可能漏召回。

registry 把 Schema 按名称排序, 只保证列表顺序稳定, 不保证别的。

### 参数保真与字符串复制

fidelity_check 把带中文引号、ASCII 引号、换行和路径 docs/a b.md 的参数做一次 JSON 往返, 逐字保留 (roundtrip 为 True)。

自动把中文引号换成英文引号后和原文不再相等 (normalized 为 False): 文件路径、引号、反斜杠和换行都可能改变动作含义。

精确编辑必须保留原字符串, 不把可读性整理偷偷混进参数。

### 读代码的顺序

- `demo.py:Spec`
- `demo.py:Discover`
- `demo.py:prompt_trials`

## 运行

```bash
python -m llm_agent.m26_tool_discovery.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `flat, discovery, poisoned` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

DiscoveryPolicy 是规则模型, 只读工具标签和返回值决定下一步; prompt_trials 是固定语法解释器, 1 / 3 / 4 这三个分数不是真实模型的提示词胜率。

没有联网 MCP、向量工具搜索; 代价只按可见 Schema 个数计 (少于 10 个算达标), 没有算 token。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

已经有 MCP tools/list, 为什么还要主动发现?

<details><summary>答案</summary>

启动时列全工具不等于按任务加载。协议枚举与模型可见范围是两件事。运行中发现能力后仍须经过版本和权限检查。

</details>
