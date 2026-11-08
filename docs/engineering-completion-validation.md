# 工程补全验证记录

日期：2026-11-08。修订内容见 [补全记录](engineering-completion.md)，机器可读结果见 [验证 JSON](engineering-completion-validation.json)。

| 检查 | 结果 |
| --- | --- |
| Python 快速集 | 271 通过，39 个 slow 用例未运行 |
| 收尾后的专项边界测试 | 32 通过；与快速集有重叠，不相加 |
| 工程实验 | 6 组、31 条件全部实际运行；末版导出与重新执行一致 |
| 旧知识 / 研究实验 | 7 组 30 条件、25 组 76 条件重新执行导出；源码摘要与最终源码一致 |
| 内容清单 | G1–G6、主讲入口、README、自测、关联链接一致；原图 45 主题/90 条原差距映射仍完整 |
| ai-agent-book 对照 | 287 个标题、10 章本地入口及固定书版本检查通过 |
| 工程网页 | 186 个条件/主题/屏宽组合通过；MCP 协议收尾后定点复测 48 组合通过 |
| 原研究网页 | 456 个条件/主题/屏宽组合通过，含四组迁移后的入口 |
| 全站计数 | 113 个章节入口，175 个实际实验入口，与目录一致 |
| 三级导航 | 113 路由 × 明暗主题；编号、缩进、文字边界与对比度通过 |
| 侧栏 / 手机导航 | 侧栏 200/260/480px、手机 320/390/768px；键盘、折叠、浏览器返回通过 |
| 外描边与交互 | 113 路由、678 个视口/主题组合；键盘边缘与语音时间轴交互通过，无裁切报告 |
| 关联链接 | 12 个页面的键盘跳转与返回通过 |
| 源码路径排版 | 修复长路径裁切；6 个页面的 320px 文字边界通过，桌面与手机截图人工复核 |
| 静态内容 | 393 道题、319 个源码引用、1473 个公式、102 个正文演进链通过 |
| 构建与分包 | 生产构建通过；入口 JS 187.87 kB，各阶段正文与题库仍按需加载 |

网页检查未发现 JS 或 HTTP 加载错误。测试使用的 HTTP 服务仅监听 loopback；全量 Python 快速集显式移除模型 API key，未调用外部模型服务。

## 复现

```bash
pytest
pytest tests/test_engineering_completion.py
python scripts/check-engineering-coverage.py
python scripts/check-research-coverage.py
python scripts/check-agent-book-coverage.py --book /data00/ai-agent-book
```

先启动 Web，再运行：

```bash
python web/scripts/check-research-completion-ui.py --dataset engineering
python web/scripts/check-research-completion-ui.py --conditions-only
python web/scripts/check-curriculum-links-ui.py
python web/scripts/check-navigation-ui.py
python web/scripts/check-lab-outlines-ui.py
```

Web 静态与构建检查在 `web/` 内执行：

```bash
npm run check:quiz
npm run check:sources
npm run check:math
npm run check:evolution
npm run build
npm run check:bundle
```

这些结果验证教学实现、数据与页面交互；没有执行真实 GPU、OAuth 身份提供方、通用 PDF/OCR 服务或生产流量实验。对应边界保留在各模块 README 与补全记录中。
