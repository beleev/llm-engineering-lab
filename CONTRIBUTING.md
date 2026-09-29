# 参与贡献 / Contributing

欢迎修复代码、改进讲解和增加实验。中文或英文 issue / PR 都可以。提交时说明读者遇到了什么问题、如何复现，以及改动验证了什么。

Contributions in Chinese or English are welcome. Include a reproducible example and explain what the experiment demonstrates. Keep examples small enough to run on a CPU, and distinguish measured results from estimates.

## 本地运行

```bash
pip install -e ".[dev]"
python main.py
pytest
pytest -m slow

cd web
npm ci
npm run dev
```

`pytest` 会自动发现 demo 和推理脚本，训练脚本由 `pytest -m slow` 运行。测试前取消设置 `ANTHROPIC_API_KEY`：`m15_claude_api` 在 SDK 和 key 都存在时会调用真实 API。只改一个实验时，先运行对应的 `python -m …demo`。

## 改代码和讲义

- 先查 `core/`、`layers/`、`training/` 和相邻模块，复用已有实现。
- 行为变化要有能失败的断言：例如与基线输出一致、梯度一致、留出集指标达到预期。不要只用“能运行”或“loss 下降”证明结论。
- 公式、运行输出和 README 中的数字一起核对。模拟的通信、显存或延迟明确写成模拟或估算。
- 文件名统一为 `README.md`。包首页使用“概览 → 运行 → 模块与阅读顺序 → 实现说明 → 边界”；教学章使用“直觉 → 核心原理 → 运行 → 运行后应该看到什么 → 与真实系统的差距 → 常见误区 → 自测题”。
- 章节配图使用真实实验台截图，保存在 `docs/screenshots/`，写明图展示什么，并链接到对应在线章节。界面或实验行为变化后更新截图。

## 改网页

实验台约定见 [web/LABS.md](web/LABS.md)。章节正文和题库按阶段加载；目录与阅读时长由构建插件从内容中提取。不要在首页重新导入全部正文。

```bash
cd web
npm run check:sources
npm run check:math
npm run check:quiz
npm run check:evolution
npm run build
npm run check:bundle
```

核对修改过的页面、跨阶段跳转和直接刷新。实验台需支持键盘操作，390 px 宽屏幕上页面不能横向溢出。更新分享卡片时保留 1200 × 630 的 PNG，设计源文件是 [docs/share-card.html](docs/share-card.html)。

## 提交 PR

标题说明具体变化。正文包含复现方式、修改后的行为、运行过的检查；网页变化附截图。较大的新主题先开 issue 说明要教什么、如何验证，再增加代码。不要提交密钥、环境文件、训练产物或 `web/dist/`。

## 发布

版本号同步更新 `pyproject.toml`、`llm_models/__init__.py`、`web/package.json`、`web/package-lock.json` 和 `CITATION.cff`。发布说明放在 `docs/releases/`。先等目标提交的 Tests 工作流通过，再为该提交创建 tag 和 GitHub Release，最后确认 Pages 部署成功。
