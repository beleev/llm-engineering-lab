# 怎么给教程加一个交互实验台

目标：读者**拖一下、点一下，就看见这个技术点为什么存在**。一个 lab 只讲一件事。

范例：`src/components/labs/PipelineLab.vue`（约 130 行，GPipe vs 1F1B）。先读它再动手。

## 加内容只需要新增文件，不改任何注册表

| 想加什么 | 新增 / 编辑哪个文件 | 自动发生什么 |
|---|---|---|
| 实验台 | `src/components/labs/<Name>Lab.vue` | 文件名即 lab 名，自动注册 |
| 把 lab 挂到某章 | `src/data/labmap/<stage>.js` → `{ 'route-name': ['NameLab'] }` | 该章出现"动手实验台"一节 |
| 新章节 | `src/data/topics/<stage>.js` → `{ stage, chapters, pages }` | 路由 `/stage/name`、侧栏、上一章/下一章全部自动生成 |
| 章末自测 | `src/data/quiz/<stage>.js` → `{ 'route-name': [{ q, options, answer, why }] }` | 章末出现自测；全对后侧栏打 ✓ |
| 术语 | `src/data/glossary/<stage>.js` → `[{ term, aka?, stage, oneliner, number?, route }]` | 出现在"术语速查"页 |
| 真源码 | 章节 page 里写 `source: ['llm_x/path/file.py:函数或类名']` | 构建期直接读 Python 文件，不会和仓库漂移 |

`<stage>` 取 `basic | models | train | finetune | infer | agent`。章节字段说明见 `src/data/topics/index.js` 顶部注释，现有章节写法见 `src/data/topics/<stage>.js`——所有正文都住在那里，`models.js` 只留结构性数据（阶段、目录、模块表、时间轴）。

**不要手抄 Python 代码到页面里**——用 `source`。`snippet` 只放刻意简化过的骨架/伪代码。

## 工具箱（都已写好，直接用）

```js
import LabFrame from '@/components/lab/LabFrame.vue'     // 外框: title / sub / module / run / challenge; 插槽 controls · default · stats · footer
import LabSlider from '@/components/lab/LabSlider.vue'   // <LabSlider v-model="x" label="…" :min :max :step unit format />; 传了 format, 读屏器读的也是 format 后的文字
import StepPlayer from '@/components/lab/StepPlayer.vue' // 开头 / 上一步 / 播放·暂停 / 下一步 / 拖动时间轴 / 播放速度
import { useStepper } from '@/composables/useStepper.js' // const s = useStepper(() => frames.value.length)
import { useDrag } from '@/composables/useDrag.js'       // 指针拖拽, 鼠标+触屏; SVG 内自动换算成 viewBox 坐标
import { mulberry32, randn, softmax, entropy, clamp, lerp, sum, range, argmax, fmtNum, fmtBytes, heat } from '@/utils/labmath.js'
```

共用样式在 `src/styles/main.css` 的 "Lab kit" 一节：`.lab-controls .row`、`.ctl`、`.lab-stats .kv > b(.good|.bad)`、`.lab-note`、`.cells` + `.cell(.on|.hot|.ok|.bad|.dim)`、`.draggable`。lab 自己的 `<style scoped>` 只写这个 lab 特有的东西。

## 写法约定

1. **状态在 `ref`，计算在 `computed`，模板只管画。** 模拟逻辑写成纯函数式的 `computed`，滑杆一动全部重算。不要在模板里写逻辑。
2. **模拟必须和 Python 模块算的是同一件事**，关键数字要对得上（例：PP=4、M=8 时 1F1B 峰值是 `[4,3,2,1]`）。拿不准就去跑对应的 `python -m …demo`。
3. **随机数用 `mulberry32(seed)`**，给一个"换一组"按钮改 seed。拖滑杆时图形不能乱跳。
4. **至少两种交互**：滑杆和按钮之外，再给一种直接操作——拖动图上的点、点击格子切换状态、悬停高亮关联元素、步进播放。标准是"读者的手放在被解释的那个量上"。
5. **右侧 stats 给 2–4 个会变的数字**，好的变绿 `.good`、坏的变红 `.bad`。数字比形容词有说服力。`.good` / `.bad` 跟着数值比较走, 不跟着按钮模式走: 写 `:class="a < b ? 'good' : 'bad'"`, 不写 `:class="mode === 'x' ? 'good' : 'bad'"`。两种模式的数字一样时, 颜色也应该一样。
6. **每个 lab 配一个 `challenge`**：先让读者预测，再展开看答案。答案要点破这个技术的本质取舍。
7. **颜色只用 CSS 变量**（`--accent --left --eye --right --warn --danger --text-*` …），明暗主题才都能看。不要写死色值。不要给会随滑杆变化的颜色加 `transition`（颜色滞后于数字会出现一瞬间的错误画面）。
8. **可访问性**：可点击的东西用 `<button>`；SVG 里的可交互节点加 `tabindex="0"`、`role`、`@keydown.enter`；滑杆用 `LabSlider`（自带 label 关联）。
   - SVG 根节点包着可交互的子节点时, 根节点用 `role="group"` 加 `aria-label`。`role="img"` 会让读屏器把整张图当成一张图片, 读不到里面的滑杆和按钮。纯展示的 SVG 保持 `role="img"`。
   - 要一个长得像按钮的链接, 写 `<router-link class="btn">`, 不要在链接里套 `<button>`。
9. **窄屏**：宽图放在默认插槽里（`.lab-viz` 自带横向滚动）；不要给容器写死像素宽度。390px 宽时页面不能出现横向滚动条。
10. 一个 lab 控制在 **120–250 行**。超了说明它在讲两件事，拆开。
11. 注释用中文、短、讲"为什么"。关键的那一行用 `★` 标出来。
12. **「玩具」和「实测表」不联动时要标明。** 一个实验台里, 左边是前端现算的玩具, 右边贴的是 Python demo 跑出来的表, 拖滑杆时表不会变。在实测那一块的标题或说明里写「实测, 不随左侧变化」, 读者才不会以为滑杆坏了。

## 自测

```bash
cd web && npm run dev          # 手动点一遍
npx vite build --outDir /tmp/llm-dist --emptyOutDir   # 必须零报错 (共享卷上直接 build 到 dist/ 偶尔会因清目录失败, 与代码无关)
```

## 画图的共用件

**结构图一律用 `components/dag/DagView.vue`，不要手写 SVG 坐标，也不要在 `<pre>` 里用框线字符画。**
只写"谁接谁"，坐标交给 dagre 算。改一个节点不用重算一片坐标，折叠、展开、换方向都不会把版式挤乱。
数据格式和插槽写在组件顶部的注释里，这里只列用法：

- 能折叠的一段 → 节点标 `group`，展开时自动画框，折起来变成一个节点
- 只想圈起来、不需要折叠 → 节点标 `box`，再传 `boxes: [{ id, label, parent? }]`；`parent` 用来套框
- 节点里要放比例条、多行数值 → 用 `#node` 插槽，外框和连线还是 DagView 管
- 静态图放 `data/diagrams/*.js`；跟页面状态变的图直接在组件里写 `computed`
- 悬停、选中这类状态写进节点或边的 `active` / `dim`。这些字段只改样式, 不触发重新排版, 读者拖过的节点位置不会丢
- 同一对节点之间可以写两条同向的边, 两条都会画
- 滚轮不缩放 (图嵌在长页面里, 滚轮要留给页面)；缩放用右上角按钮, 左下角「重排」回到自动排版

**判断该不该用 DagView，只看一条：位置本身是不是信息。**

- 位置只是人手摆的接线版式 → 用 DagView，接线关系才是信息
- 位置编码了数据（时间轴、柱高、序号、角度）→ 不用。它是图表，dagre 为了减少交叉会打乱顺序，把信息弄丢
- 有回边的（状态机）→ 不用，它不是 DAG

另外还有三个共享件。目前只有 `components/InspectorPanel.vue` 在用它们, 而 InspectorPanel 只挂在「Block 组装器」一页 (`views/Blocks.vue`)。别的章节要画计算流时可以直接拿来用：

| 组件 | 干什么 |
|---|---|
| `components/FlowDiagram.vue` | 计算流。每一步自带一条与张量元素数成比例的 `SizeBar`，所以 `[B,H,T,T]` 随 T 长出来的样子是看得见的。传 `active-param` 可以让用到某个权重的步骤高亮 |
| `components/SizeBar.vue` | 一条正比于张量大小的横条，batch 维不计，只比形状本身 |
| `components/ParamsTable.vue` | 权重表。悬停、键盘聚焦或点一下某一行, `@hover` 抛出权重名，配合 `FlowDiagram` 的 `active-param` 做两栏联动 |

画结构图时先问一句：**这张图有没有把"大小"画出来？** 两个形状写出来一样长，但元素数差一百倍——这种差别只有画成面积或长度才会被看见。

## 划重点：主干 / 扩展 / 冲刺

`src/data/tiers.js` 把每一章分成三层，**只影响阅读建议，不影响内容和可达性**：

| 层 | 标记 | 含义 |
|---|---|---|
| 冲刺 | ★ | 12 章。只有一天就读这些，每个阶段不读就接不上下一阶段的那几页 |
| 主干 | ● | 42 章（含冲刺）。完整课程主线 |
| 扩展 | ○ | 43 章。深水区与分支；跳过不影响主线 |

合计 86 章 = 主干 42 + 扩展 43 + 速查 1 (总览对照表)。主线总览是序章, 不计入。

分层出现在四个地方：侧栏的档位筛选、面包屑的标记、`/fast-track` 速成路线页、翻章 (「上一章 / 下一章」和键盘 ← → 只在当前档位的章节之间走)。侧栏和速成路线页的档位是同一个值, 存在 `useProgress().state.level`。新增章节默认归为扩展——主线是要守住的，加内容不该让它变长。要改归类就改 `tiers.js` 里的两个数组，其余全是算出来的。

**章内划重点**：每个章节的 `points` 里，给最该带走的那一条加 `key: true`，页面会渲染一个「重点」徽章。一章只标一条。

```js
points: [
  { title: 'cache 是手写 autograd tape', body: '…', key: true },
  { title: '形状先行', body: '…' },
]
```

## 写文案的规矩（说人话）

这是教程，不是文档。读者读一遍就要懂。

1. **先说具体的，再说抽象的。** 先讲发生了什么，再给它起名字。
2. **拆掉名词堆。** 「每个 forward 返回反向需要的输入、权重、归一化统计」→ 谁把什么交给谁，写成有动词的句子。
3. **一个数字胜过一个形容词。** 「显著更省」→「省 56.9×」。数字必须来自真实跑出来的 demo。
4. **先说为什么疼，再说怎么治。** 每个技术都是因为上一代出了问题才存在的。
5. **短句。** 不要「值得注意的是」「我们可以看到」「综上所述」。
6. `tldr` 是一句能背下来的话；`question` 是读者真会问的问题；`subtitle` 是读完这页你能做什么。
7. **结论按实测写, 不挑好看的写。** 实测不符合流行说法就照实写，并说明是在什么规模下测的。

### 版式：短段落 + 要点

读者扫一眼就要能抓住结构。一段只讲一件事，并列的东西列成要点。

- **段落**：不超过 3 句、约 60 字。超了就拆。
- **要点**：对比 (A 怎样 / B 怎样)、步骤、条件分支、代价清单，一律列成要点。每条一两句，不超过约 70 字。
- **顺序**：先结论，再机制，最后代价或边界。
- **只拆不删**：数字、公式、术语、条件、因果一个都不能丢。改完逐项核对原文里的数字还在不在。

**字符串字段**（`body` / `tldr` / `subtitle` / `question` / `takeaway` / `why` / `description` / 实验台的 `sub`、`challenge.ask` 和 `challenge.answer`）
由 `components/Prose.vue` 渲染。写法：

```js
body: '结论一句。\n- 关键词: 第一条。\n- 关键词: 第二条。\n收尾一句。',
```

- 换行 = 分段；`- ` 开头的连续行 = 要点列表。
- 要点以「20 字符以内的关键词 + 冒号」开头时，关键词自动加粗。
- 不要在一条要点中间换行：续行会变成独立段落，把列表截断。
- 模板里的静态属性（`sub="..."`）直接写真换行即可，Prose 按行切段。
- 新加一个字符串字段时，确认它的渲染处用的是 `<Prose :text="..." />`，不是 `{{ }}`。

**模板里的文字**（手写页面、实验台读数区）用这三个类，样式和 Prose 一致：

- 多段导语：`<div class="lead-group">` 里放多个 `<p>` / `<ul class="pts">`，代替一个很长的 `<p class="lead">`。
- 要点列表：`<ul class="pts">` / `<ol class="pts">`，关键词用 `<b>…:</b>`。
- 实验台读数区的多段说明：`<div class="lab-note">` 包多个 `<p>`，仍是一条左边框。

## 公式：KaTeX

数学公式一律写成 LaTeX，交给 KaTeX 排版，不要用 Unicode 拼（`√d_k`、`Σ`、`ᵀ`、`x₀`）。

- **行内** `$…$`，**独立成行** `$$…$$`（在 Prose 里要单独占一行；多行推导放进一个 `$$\begin{aligned}…\end{aligned}$$`）。字面的美元符写 `\$`。
- **在哪能用**：经过 `<Prose :text>` 或 `<Tex :text>` 渲染的字符串。`{{ }}` 直接插值的地方不会排版，要换成 `<Tex :text="…" />`。
- **JS 字符串里反斜杠写两个**：`'$\\sqrt{d_k}$'`。只写一个的话，`\s` 会被 JS 吃掉变成 `sqrt`，`\f` `\n` `\t` 会变成控制字符，KaTeX 还不报错。模板静态属性（`text="$\sqrt{d_k}$"`）里写一个。
- **两个不报错的坑**：`%` 在 KaTeX 里是注释，会把后面的公式静默吞掉，要写 `\%`（JS 字符串里 `\\%`）；单引号字符串里写撇号 `W'` 会截断字符串，用 `W^\prime`。模板静态属性里的 `<` `>` 写 `\lt` `\gt`。
- **写法**：函数名用 `\mathrm{softmax}`、`\log`、`\exp`；转置 `^\top`；多字母下标 `W_{\text{down}}`、`\mathcal{L}_{\text{DPO}}`；公式里的中文放 `\text{…}`。
- **哪些不用 KaTeX**：代码（`snippet`、`sourceRows.code`、`<pre class="code">`）、代码标识符（`num_kv_heads`、`pos // bs`）、张量形状（`[B,T,D]`）、SVG `<text>` 里的标签。它们保持等宽字体和 Unicode。

`npm run check:math` 会把 `data/` 下所有公式用 KaTeX 解析一遍，并查出漏掉的反斜杠。`.vue` 里的公式在源码里分不出来，要在浏览器里看有没有红色的 `.katex-error`。

## 出章末自测题

每道题 4 个选项。读者不懂内容时，不能靠「挑最长、最具体的那个」蒙对。

- **干扰项写成真实的误解**，和正确项一样具体：把因果说反、混淆相邻概念（GQA 和 MLA、forward 和 reverse KL）、机制对但数字错、只在另一种条件下成立、流行说法但本章实测推翻了它。
- **不写一眼就假的选项**（「numpy 不支持 float32」「为了好看」）。
- **正确项不带解释性的尾巴**，那部分写进 `why`。
- 干扰项必须**确实是错的**：对照本章正文和 Python 源码。
- `npm run check:quiz` 已进 CI：正确项比次长选项长 30% 以上就报错。
