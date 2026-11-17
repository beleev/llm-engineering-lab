# 程序化行动 · 规则、适配器与生成界面

[![程序化行动 · 规则、适配器与生成界面](../../docs/screenshots/agent-programmatic-actions-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/programmatic-actions)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/programmatic-actions)

## 直觉

把折扣规则写成代码, 跑出来的数就一定对吗?

- 程序化行动指让模型把规则写成代码交给解释器算, 而不是在文字里心算。代码只保证可重复: 写错的折扣公式 total * (1 - percent) 每次都稳定算出 -900。
- 三道验收各管一件事:
- 业务用例 "期望 90" 验规则对不对。
- 受限 AST (只放行数字、变量、四则的语法树白名单) 拒掉 __import__ 这类调用。
- html.escape 和 SVG 文本节点把用户字符串里的 <img>、<script> 变成普通文本。

## 核心原理

### 代码是可执行的思考, -900 也执行得很稳

算术、表格处理和复杂约束交给解释器算, 比让模型反复用自然语言心算稳定: 同一个表达式代同样的数, 每次结果都一样。

但稳定不等于正确。demo 的 naive 模式把折扣规则写成 total * (1 - percent), 把 total = 100、percent = 10 代进去: 1 - 10 = -9, 100 × (-9) = -900。每次跑都是 -900, 解释器不会报错, 因为这个式子在语法上合法。

validated 模式把式子改成 total * (1 - percent / 100): 1 - 0.1 = 0.9, 得到 90.0。

判断哪个对的不是解释器, 是 "期望 90" 这条业务用例: demo 的 trace 里 "业务验收" 一步拿算出来的数和 90 比, naive 挂在 "期望 90, 实际 -900"。边界用例省不掉。

### 工具、执行器、Skill 与适配器

让模型 "用代码行动" 有三种载体:

- 专用工具: 接口窄而稳定, 比如 calculate 只做四则运算, 能做的事一眼看得完。
- 通用执行器: 什么代码都能跑, 组合能力强但风险大, 本章不提供。
- Skill: 一份说明, 告诉模型何时、怎样用前两者。

适配器 adapt 管的是外部数据进来时的格式转换。外部接口返回 {order_id, amount_cents} (金额以分为单位), 内部要的是 {id, amount} (以元为单位): order_id 1004 变成字符串 "1004", amount_cents 1999 除以 100 变成 19.99。它先核对字段集合必须正好是 {order_id, amount_cents}, 而且 amount_cents 必须是 int; 多一个字段、少一个字段, 或类型不对, 直接抛 "响应 Schema 不匹配" (Schema 指字段结构的约定)。外部接口悄悄改了格式 (接口漂移), 适配器会当场报错, 不会静默接受。

### 生成 UI 与媒体: 先转义再拼

模型生成界面时, 用户提供的字符串和界面结构要分开拼: 字符串先转义, 再塞进结构里。

make_ui 收到用户字符串 <img src=x onerror="run()"> (一个会执行脚本的图片标签), 先用 html.escape 把 < > " 这些字符换成 &lt; &gt; &quot;, 再放进 <h2> 里。输出里找不到 <img, 浏览器会把它当文字显示, 不会当标签执行。

make_svg 用 ElementTree (标准库的 XML 工具) 建 <svg><text>, 把 <script>not executable</script> 作为 .text 文本节点放进去。序列化时它变成 &lt;script&gt;, 还能被 ET.fromstring 重新解析成合法 XML。

三种模式 (naive、validated、rejected) 这两项检查都通过, 它们和算式对不对无关。媒体用程序 (SVG 这类结构化表示) 生成, 便于编辑和复验, 但不意味着生成的界面可以任意执行脚本。

### 自举也需要外部约束: AST 白名单

自举 (Agent 自己写自己的适配器或测试) 可以, 但它不能改评分标准再宣称进步: "期望 90" 这条用例写在 experiment 外面, 模型生成的表达式改不到它。

代码能不能跑由 calculate 的白名单决定。它先用 ast.parse 把表达式解析成 AST (抽象语法树, Python 把代码拆成的树状结构, 每个运算、数字、变量名是一个节点), 然后只放行三类节点: 数字常量、已知变量名、四则运算 (+ - * /)。节点超过 40 个算超预算; 除零和溢出都抛 ValueError。整个过程没有 eval, 不会真的把字符串当代码执行。

rejected 模式的表达式是 __import__('os').system('echo unsafe'): 解析后遇到的是函数调用 Call 节点, 不在白名单里, 当场被拒, amount 为 None。

受限 AST 只是最便宜的一层。要跑通用生成代码, 仍需要单独的隔离执行环境。

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

它展示的是表示方式: 规则、适配器、界面各自怎么验, 不衡量通用代码生成的质量。40 个节点的预算和只许四则是教学取值, 真实业务规则装不进去。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

把错误业务规则改成确定性代码, 会怎样?

<details><summary>答案</summary>

错误会稳定复现, 仍需规则验收。可重复执行不是语义正确性。测试既要看程序行为, 也要验证规则适用范围。

</details>
