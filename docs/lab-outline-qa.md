# 实验台布局与焦点检查

实验台使用可滚动容器容纳宽图，选中描边和键盘焦点应完整可见。页面在窄屏下允许源码路径、标题和运行命令换行。

## 布局约定

- `.lab-viz` 与 `.lab-scroll` 为外描边留出 6px 空间，覆盖 3px outline 与 3px offset；负 margin 抵消内边距，保持内容起点和图表宽度。
- 靠近 SVG viewBox 边缘的控件使用内描焦点框，SVG 保持裁剪与原坐标换算。
- SFT 固定列网格使用实际内容宽度，滚动区末尾保留可见空间。
- 主题目录与后续卡片保持 36px 间距；展开与收起均使用同一布局规则。

## 运行

先启动本地 Web，再运行：

```bash
python web/scripts/check-lab-outlines-ui.py
python web/scripts/check-lab-outlines-ui.py --focused
python web/scripts/check-lab-outlines-ui.py --layout-only
python web/scripts/check-navigation-ui.py --base http://127.0.0.1:5174/llm-engineering-lab/
```

描边检查默认输出 `/tmp/lab-outline-qa.json`。脚本使用浏览器 computed style、元素矩形和各级滚动裁剪边界，检查选中状态与实验台首尾控件获得键盘焦点的状态。正常位于横向视口之外的内容不计为裁剪错误；SVG 在布局稳定后读取样式。

语音交互另检查条件切换、事件选中、时间轴 Home 和下一步操作。导航检查覆盖章节编号、缩进、对比度、侧栏宽度与移动端抽屉。

## 范围

几何检查仅针对 border box 已完整进入滚动视口的控件，不覆盖所有参数组合。像素差异、屏幕阅读器、性能与 WCAG 合规需要各自的验证方法。
