# 视觉操作 · 截图、定位与核对

[![视觉操作 · 截图、定位与核对](../../docs/screenshots/agent-visual-grounding-coverage.png)](https://beleev.github.io/llm-engineering-lab/#/agent/visual-grounding)

[Agent 目录](../README.md) · [交互实验](https://beleev.github.io/llm-engineering-lab/#/agent/visual-grounding)

## 直觉

我截图、找到绿色按钮中心 (29, 17)、点下去, 每一步都没错, 怎么打开的是另一个订单?

同一个绿色按钮, 截图之后和红色按钮换了位: 按旧坐标 (29, 17) 点开的是 other, 刷新后 (29, 52) 才是 target。点击只负责打开, committed 只在核对 opened == target 之后为真, 三种条件 wrong_commit 都是 False。

## 核心原理

### 从像素找到目标

截图是 80 × 80 的像素栅格, 上面两个 40 × 16 的按钮: 绿色 (20, 180, 80) 是目标订单, 红色 (180, 40, 60) 是另一个订单。locate 取所有绿色像素的包围盒中心: 目标在 (10, 10) 时得 (29, 17), 在 (10, 45) 时得 (29, 52)。

截图里一个绿像素都没有就抛 ValueError, 不猜坐标。browser.py 用 Chromium 真截图、真鼠标点击, 同一个 locate 读真实像素, 所有网络请求被阻断。

### 观察和行动之间环境会变

render(shift=True) 把两个按钮上下换位, 模拟截图之后页面重排。stale 条件按旧截图的 (29, 17) 点下去, hit 返回 other: 坐标没算错, 算的是上一帧。

refresh 条件点击前重新截图, 定位到 (29, 52), 打开 target。刷新只能保证截图那一刻, 不能保证点击瞬间页面没有再变。

### 把导航和副作用分开

三种条件里点击都只做一件事: 打开详情。committed 只在 opened == target 时为真: stale 打开 other, committed=False; verify 第一次也打开 other, 重截重点 (29, 52) 后才 True, observations=2。

三种条件 wrong_commit 全是 False: 误点能回退, 错误提交可能不可逆。需要事务、授权和人工确认的动作, 不能只看按钮上的字。

### 不局限于浏览器

桌面和手机一样要有感知 (截图)、动作空间 (点击 / 输入)、焦点和输入状态; 这里的 80 × 80 栅格加 hit 函数就是最小的一套。

音视频 GUI 还要在时间轴上采样; 世界模型可以预测一次动作的后果, 但预测仍要拿环境反馈核对, 和 verify 条件重截一次是同一件事。

### 读代码的顺序

- `demo.py:locate`
- `demo.py:hit`
- `demo.py:experiment`

## 运行

```bash
python -m llm_agent.m35_visual_grounding.demo
```

默认只用标准库与本仓库模块, 不访问外网。

真实浏览器版本是可选开发验证, 依赖 `playwright` 与 `pillow`:

```bash
python -m pip install playwright pillow
python -m playwright install chromium
python -m llm_agent.m35_visual_grounding.browser
```

它在 Chromium 中渲染本地页面、读取截图像素、点击并核对对象; 所有网络请求被阻断。

## 运行后应该看到什么

依次运行 `stale, refresh, verify` 三种条件, 输出结果与逐步轨迹。`main()` 对预期的成功和失败分别断言; 失败对照不是程序故障。网页可选择配置、播放或点击事件, 显示 Python 已运行记录, 不在浏览器里执行训练或工具。

## 与真实系统的差距

定位靠一种固定颜色, 不是通用视觉语言模型; 两个同色元素会被算成一个包围盒的中心。

真实浏览器夹具本地渲染、不联网, 要装 playwright 和 pillow。不声称能操作任意站点、桌面或手机。

## 常见误区

“有相同概念的教学实验”不等于“复现完整论文或具备通用产品能力”。对照只隔离本章说明的变量; 其他假设变化需重新测。

## 自测题

刷新截图后点中按钮, 何时才可以确认业务提交?

<details><summary>答案</summary>

核对当前对象和动作授权之后。几何定位不能证明业务对象与权限。导航后的核对是独立验收步骤。

</details>
