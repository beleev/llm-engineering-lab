# 验证指南

检查分为 Python 行为、实验回放一致性、网页静态内容和浏览器交互。默认实验在 CPU 上运行；通过这些检查不表示已验证真实模型服务、GPU 吞吐或生产系统。

## Python 行为

在仓库根目录安装 `pip install -e ".[dev]"` 后运行：

```bash
env -u ANTHROPIC_API_KEY OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
# 慢速训练单独运行
env -u ANTHROPIC_API_KEY OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -m slow
```

默认 pytest 排除 slow。API key 在测试进程中取消设置，防止可选在线示例访问模型服务；线程数限制用于减少小模型的调度开销。

## 实验与网页证据

```bash
python scripts/check-agent-coverage.py
python scripts/check-research-coverage.py
python scripts/check-engineering-coverage.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python web/scripts/export-agent-coverage.py --check
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python web/scripts/export-knowledge-completion.py --check
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python web/scripts/export-research-completion.py --check
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python web/scripts/export-engineering-completion.py --check
```

清单检查核对实验、文档、源码符号、运行条件、自测与章节归属。导出器检查重新执行 Python，并比较指标、事件和源码摘要。修改源码或实验清单后，去掉 `--check` 重新导出，核对模块 README 的数字，再运行检查。

## Web 静态内容与构建

在 `web` 目录运行：

```bash
npm ci
npm run check:sources
npm run check:math
npm run check:quiz
npm run check:evolution
npm run check:consolidation
npm run build
npm run check:bundle
```

源码检查覆盖页面引用的 Python 符号，数学检查解析公式，自测检查题目结构及选项长度，演进链检查技术讲解的逻辑。目录检查核对主讲入口、别名、段落和学习状态，分包检查约束正文与题库的按需加载。

## 浏览器交互

安装可选开发工具并在独立终端运行 Web 服务：

```bash
python -m pip install playwright
python -m playwright install chromium
# 独立终端，在 web 目录运行：npm run dev -- --host 127.0.0.1 --port 5174
python web/scripts/check-navigation-ui.py --base http://127.0.0.1:5174/llm-engineering-lab/
python web/scripts/check-consolidation-ui.py
python web/scripts/check-curriculum-links-ui.py
python web/scripts/check-agent-ui.py --base http://127.0.0.1:5174/llm-engineering-lab/
python web/scripts/check-agent-coverage-ui.py --base http://127.0.0.1:5174/llm-engineering-lab/
python web/scripts/check-knowledge-completion-ui.py
python web/scripts/check-research-completion-ui.py
python web/scripts/check-research-completion-ui.py --dataset engineering
python web/scripts/check-lab-outlines-ui.py
```

这些脚本覆盖条件切换、指标读数、源码展开、挑战题、键盘、深链接、前进后退、侧栏、窄屏和明暗主题。实验描边检查的范围见 [布局与焦点检查](lab-outline-qa.md)。真实 Chromium 视觉定位示例单独运行 `python -m llm_agent.m35_visual_grounding.browser`，并按模块 README 安装可选依赖。

检查结果以实际运行输出为准。几何检查不替代像素审查、屏幕阅读器审计或真实性能测试。
