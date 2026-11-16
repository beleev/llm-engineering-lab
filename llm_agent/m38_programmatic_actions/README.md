# 程序化行动 · 规则、适配器与生成界面

[![程序化行动 · 规则、适配器与生成界面](../../docs/screenshots/agent-programmatic-actions-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/programmatic-actions)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/programmatic-actions)

## 直觉

把折扣规则写成代码, 跑出来的数就一定对吗?

把规则写成代码只保证可重复: total * (1 - percent) 每次都算出 -900。期望值 90 验规则, 受限 AST 拒掉 __import__, html.escape 和 SVG 文本节点把 <img>、<script> 变成文本。

## 核心原理

### 代码是可执行的思考, -900 也执行得很稳

算术、表格处理和复杂约束交给解释器, 比让模型反复用自然语言算稳定。但稳定不等于正确: naive 模式把折扣写成 total * (1 - percent), total = 100、percent = 10 代进去跑出 -900, 每次都是 -900。

validated 模式改成 total * (1 - percent / 100), 得到 90.0。判断哪个对的不是解释器, 是 "期望 90" 这条业务用例; 错误规则写成代码只会稳定地执行错误, 边界用例省不掉。

### 工具、执行器、Skill 与适配器

- 专用工具: 稳定窄接口, calculate 只做四则。
- 通用执行器: 组合能力强但风险大, 本章不提供。
- Skill: 说明何时、怎样用前两者。

适配器 adapt 把外部 {order_id, amount_cents} 转成内部 {id, amount}: 1004 / 1999 → "1004" / 19.99。多一个字段、少一个字段, 或 amount_cents 不是 int, 直接抛 "响应 Schema 不匹配", 不静默接受接口漂移。

### 生成 UI 与媒体: 先转义再拼

数据和结构分开拼: make_ui 先 html.escape 用户字符串 <img src=x onerror="run()"> 再塞进 <h2>, 输出里找不到 <img; make_svg 用 ElementTree 文本节点放 <script>not executable</script>, 序列化成 &lt;script&gt;, 还能被 ET.fromstring 重新解析。

三种模式这两项检查都通过, 和算式对不对无关。媒体用程序表示便于编辑与复验, 但不意味着生成的界面可以任意执行脚本。

### 自举也需要外部约束: AST 白名单

Agent 可以写自己的适配器或测试, 但不能改评分标准再宣称进步: "期望 90" 这条用例在 experiment 外面, 表达式改不到它。

calculate 只接受数字、已知变量和四则运算, AST 节点超过 40 个算超预算, 除零和溢出都抛 ValueError。rejected 模式的 __import__('os').system('echo unsafe') 在 Call 节点处被拒, amount 为 None。

通用生成代码仍需真正的隔离执行环境, 受限 AST 只是最便宜的一层。

### 读代码的顺序

- `demo.py:calculate`
- `demo.py:adapt`
- `demo.py:make_ui`
- `demo.py:make_svg`

## 运行

```bash
python -m llm_agent.m38_programmatic_actions.demo
```

默认只用标准库与本仓库模块, 不访问外网。

## 运行后应该看到什么

依次运行 `naive, validated, rejected` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

demo 只做受限四则表达式与本地 HTML/SVG 字符串, 不执行任意 Python、JavaScript 或第三方 Skill 脚本。

它展示表示方式: 规则、适配器、界面各自怎么验, 不衡量通用生成质量。40 个节点的预算和只许四则是教学取值, 真实业务规则装不进去。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

把错误业务规则改成确定性代码, 会怎样?

<details><summary>答案</summary>

错误会稳定复现, 仍需规则验收。可重复执行不是语义正确性。测试既要看程序行为, 也要验证规则适用范围。

</details>
