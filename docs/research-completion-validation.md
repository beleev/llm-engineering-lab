# 研究方向补充：验证记录

范围：25 组新增 CPU 实验、76 个执行条件、原有章节中的网页接入、25 道自测、20 条术语，以及 README 与覆盖文档。原始审计为 `23f1c93`；新证据见 [补充覆盖与边界](research-completion.md)。

本地环境：Python 3.13.5、PyTorch 2.15.0+cpu、NumPy 2.5.4；Vite 6.4.2、无头 Chromium。Python 检查使用 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，并清除 `ANTHROPIC_API_KEY`，避免旧的可选在线 demo 发起请求。本轮没有调用模型 API、没有 GPU 或多机实测。

## Python 与内容

| 检查 | 结果 |
| --- | --- |
| `pytest`（默认排除 slow） | 239 项通过；包括所有 demo/infer 脚本、19 项新增边界回归 |
| `scripts/run-research-completion.py` | 25 组实验全部完成；包含 IPO/PEFT/在线偏好/SAE/RAG 的真实参数更新 |
| `web/scripts/export-research-completion.py --check` | 25 组、76 条条件的实际结果与源码/清单摘要一致 |
| `web/scripts/export-knowledge-completion.py --check` | 原有 7 组、30 条条件仍与源码和结果一致 |
| `scripts/check-research-coverage.py` | 原图 45 个主题、90 条原差距均有去向，文档/函数/条件匹配 |
| `scripts/check-agent-book-coverage.py` | 原书 10 章、287 个知识标题的本地入口完整 |

新增边界回归覆盖 IPO 的有效 token 与 β、检索/生成两条梯度路径、查询改写保留否定、位置偏好与证据稀释、非法证明引用、歧义 OCR 拒答、校准非法输入、遗忘与重训等价、候选测试投机。课程实验另外提供同配比、同样本批次的乱序对照；合成反馈实验固定过滤前每代总样本预算，避免把更多样本的收益全归因于真实数据锚点。

本轮修改后重新执行了新增实验；未改动的历史长训练脚本没有额外整套重跑。默认 smoke 不等同于真实数据、长周期训练或线上能力评测。

## Web 静态检查与构建

| 命令 | 结果 |
| --- | --- |
| `npm run check:sources` | 317 / 317 个源码引用有效，包含 JSON 回放引用 |
| `npm run check:math` | 1473 / 1473 个公式通过 |
| `npm run check:quiz` | 384 道自测，选项长度检查通过 |
| `npm run check:evolution` | 100 个数据驱动章节的演进链检查通过 |
| `npm run build` | 生产构建通过 |
| `npm run check:bundle` | 入口 JS 186.83 kB，六阶段正文/题库及研究回放保持按需加载 |

目录仍为六阶段、26 个主题组、111 个章节入口（含六个阶段总览），实际实验入口 182 个；术语共 248 条。实验计数按实际 LabFrame 组数计算，修正了一个回放组件在同章渲染多组实验时被少计的问题。

## 浏览器复查

新增实验检查命令：

```bash
python web/scripts/check-research-completion-ui.py --base http://127.0.0.1:5174/llm-engineering-lab/
python web/scripts/check-lab-outlines-ui.py --layout-only --output /tmp/research-outlines.json
python web/scripts/check-navigation-ui.py
```

新增实验检查涵盖全部 76 个条件，亮/暗两个主题，1440 / 768 / 320 三个宽度，共 456 个组合；逐个用键盘操作按钮，核对实际指标、原理展开、源码展开、挑战题与页面横向溢出。全站另外逐章核对实际实验数量与目录统计。课程采样对照收紧后，另复测其 18 个条件/主题/屏宽组合，通过且无 JS/HTTP 错误。

导航检查覆盖 111 个入口、两种主题，桌面侧栏宽度 200 / 260 / 480，以及手机/平板宽度 320 / 390 / 768；编号、对齐、对比度、可视区域与键盘操作通过，无 JS 错误。

全站实验边框检查复用此前修复遮线问题的检查器：111 个入口 × 3 个宽度 × 2 个主题，同时保留语音交互的条件、时间轴与下一步回归。完整结果如下；实验计数会同时包含三级章节与六个阶段总览。

| 浏览器记录 | 结果 |
| --- | --- |
| 全站实验计数 | 111 个章节 / 182 个实际实验入口与目录统计一致，无 JS/HTTP 错误 |
| 新增实验条件 | 25 组 / 76 条条件 / 456 条件主题屏宽组合通过；零 JS/HTTP 错误 |
| 全站边框与语音交互 | 111 路由 / 666 屏宽主题组合通过；零裁切、零 JS/HTTP 错误 |

## 已发现并处理的问题

- 基础阶段题库原本直接导出对象，接入增补题前先改为具名常量，再统一导出。
- 修正一道正确选项明显偏长的题目，并分散新增题目的正确选项位置。
- 校准函数拒绝非有限概率、空输入和非法桶数；IPO 拒绝没有有效 token 的回答及非法 β，避免静默 NaN。
- 复查 PEFT 后固定各模式输入样本，避免不同参数初始化消耗随机数造成训练集不同。
- 查询目录统计时等待异步实验真正挂载；不能因为延迟占位框尚未出现，就把瞬时空 DOM 当作缺少实验。

所有结果仅支持本仓库明确给出的教学配置。完整大模型、真实设备、真实语音/视觉质量、机器人接触与专业领域上线验证均未据此宣称完成。

结构化结果见 [验证记录 JSON](research-completion-validation.json)。
