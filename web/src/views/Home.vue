<template>
  <div>
    <h1 class="page-title">LLM 全栈 · 主线总览</h1>
    <div class="page-subtitle lead-group">
      <p>
        本教程将「构建一个可训练、可部署、可执行任务的完整大模型系统」拆解为<strong>六个递进工程阶段</strong>, 每个阶段分别对应仓库根目录下的一个独立子项目。
      </p>
      <p>全书正文与源码实现逐一对应, 每处机制讲解均可在仓库的 Python 文件中定位到具体实现。</p>
    </div>

    <!-- ── 新手 30 秒导览: 第一次来的人先看这里, 所以紧跟在副标题下面 ───────── -->
    <section class="section onboarding">
      <h2>快速导览 <span class="lead-inline">首次访问阅读指南</span></h2>
      <div class="grid grid-3 onboard-grid">
        <div class="card onboard-card">
          <div class="onboard-num">1</div>
          <h3>教程定位</h3>
          <p class="desc">
            这是一套<strong>源码与交互图解紧密对照</strong>的 LLM 全栈工程教程。每章均配有可独立运行的仓库源码, 点击文件链接即可直达对应实现。
          </p>
        </div>
        <div class="card onboard-card">
          <div class="onboard-num">2</div>
          <h3>先修基础</h3>
          <p class="desc">
            具备基础 Python 编程与矩阵运算知识即可开始。<span class="muted-line">起步阶段不依赖深度学习框架, 阶段 1 使用纯 NumPy 完整展开前向与反向计算。</span>
          </p>
        </div>
        <div class="card onboard-card">
          <div class="onboard-num">3</div>
          <h3>阅读方法</h3>
          <p class="desc">
            <strong>无需按目录逐章通读。</strong>建议先按背景选择阅读档位 (冲刺 {{ sprintCount }} 章 / 主干 {{ coreCount }} 章), 单章学习可分三步推进:
          </p>
          <ol class="desc pts">
            <li>调节交互实验台参数, 观察核心变量的动态响应。</li>
            <li>对照正文要点与源码映射表厘清实现细节。</li>
            <li>完成章末自测题以检验关键概念掌握程度。</li>
          </ol>
        </div>
      </div>
    </section>

    <ChapterIntro
      tldr="全书沿一条工程主线展开, 从底层算子一路搭到上层应用:
        1. llm_basic: 纯 NumPy 手写前向、反向与采样闭环。
        2. llm_models: 拆解现代主流架构的注意力、位置编码、MoE 与扩散组件。
        3. llm_train: 多卡分布式并行、显存优化与低精度训练。
        4. llm_finetune: SFT、LoRA 家族、偏好对齐与强化学习微调。
        5. llm_infer: KV Cache 分页、连续批处理、量化与投机解码。
        6. llm_agent: 工具调用、权限护栏、上下文工程与评测闭环。"
      question="面对从底层算子到上层应用的庞大链路, 不同背景的读者应当如何规划切入点与阅读顺序?"
      :goals="[
        '厘清 LLM 全栈工程的六个递进阶段及其各自解决的核心问题',
        '掌握每一阶段在代码仓库中的目录结构与核心入口文件',
        '根据自身技术背景与学习目标规划合适的阅读路径',
      ]"
      :codes="[
        { path: 'llm_basic/' },
        { path: 'llm_models/' },
        { path: 'llm_train/' },
        { path: 'llm_finetune/' },
        { path: 'llm_infer/' },
        { path: 'llm_agent/' },
      ]"
    />

    <!-- ── 接着上次学: 进度存在本机 localStorage, 回访时一步回到断点 ─────── -->
    <section v-if="resume" class="section">
      <div class="card resume">
        <div>
          <p class="resume-k">继续上次阅读</p>
          <p class="resume-t">{{ resume.label }}</p>
          <p class="desc">已读 {{ readCount }} / {{ chapters.length }} 章 · 自测全对 {{ masteredCount }} 章</p>
        </div>
        <div class="btn-group">
          <router-link :to="{ name: resume.route }" class="btn active">回到本章 →</router-link>
          <router-link v-if="nextUnread" :to="{ name: nextUnread.route }" class="btn">{{ levelLabel }}下一未读章节: {{ nextUnread.label }}</router-link>
        </div>
      </div>
    </section>

    <!-- ── 该读哪些: 章节太多, 先给一条路线。全页只有一个主按钮:
         第一次来是「从阶段 1 开始」, 回访时让给上面的「回到这一章」 ─────── -->
    <section class="section">
      <div class="card route-cta">
        <div>
          <p class="cta-k">按档位规划全书 {{ chapters.length }} 章</p>
          <ul class="cta-t pts">
            <li><b>主干 {{ coreCount }} 章:</b> 完整贯通六个工程阶段的核心链路。</li>
            <li><b>冲刺 {{ sprintCount }} 章:</b> 聚焦承上启下的关键节点, 适合单日速览。</li>
            <li><b>扩展章节:</b> 涵盖架构变体与前沿专题, 可按需选读查阅。</li>
          </ul>
        </div>
        <div class="cta-go">
          <router-link :to="{ name: 'basic' }" :class="['btn', { active: !resume }]">从阶段 1 开始 →</router-link>
          <router-link :to="{ name: 'fast-track' }" class="cta-alt">按时间规划? 查看速成路线</router-link>
        </div>
      </div>
    </section>

    <!-- ── 怎么用这本教程 (图形化路径) ───────────────────── -->
    <section class="section">
      <h2>三种阅读路径 <span class="lead-inline">按背景选择切入起点</span></h2>
      <div class="grid grid-3 path-grid">
        <div class="card path-card">
          <span class="path-tag green">🟢 入门</span>
          <h3>从底层原理起步</h3>
          <ol class="path-steps">
            <li>精读 <router-link :to="{ name: 'basic' }">阶段 1</router-link>, 用纯 NumPy 跑通最小训练与采样闭环</li>
            <li>进入 <router-link :to="{ name: 'attention' }">注意力演进</router-link> 与 <router-link :to="{ name: 'blocks' }">Block 组装</router-link>, 掌握现代主干组件</li>
            <li>对照 <router-link :to="{ name: 'compare' }">总览对照表</router-link>, 规划后续工程章节</li>
          </ol>
        </div>
        <div class="card path-card">
          <span class="path-tag orange">🟠 进阶</span>
          <h3>具备 PyTorch 经验</h3>
          <ol class="path-steps">
            <li>略过阶段 1, 直接从 <router-link :to="{ name: 'attention' }">阶段 2 · 注意力演进</router-link> 切入</li>
            <li>重点研读 <router-link :to="{ name: 'train' }">阶段 3 规模化训练</router-link> 与 <router-link :to="{ name: 'finetune' }">阶段 4 任务适配</router-link></li>
            <li>衔接 <router-link :to="{ name: 'infer' }">阶段 5 推理优化</router-link>, 贯通从训练到线上服务的主线</li>
          </ol>
        </div>
        <div class="card path-card">
          <span class="path-tag blue">🔵 工程</span>
          <h3>聚焦部署与应用</h3>
          <ol class="path-steps">
            <li>优先阅读 <router-link :to="{ name: 'infer' }">阶段 5 · 推理优化</router-link>, 掌握缓存与并发调度</li>
            <li>继续深入 <router-link :to="{ name: 'agent' }">阶段 6 · Agent 应用层</router-link>, 构建工具调用与状态闭环</li>
            <li>遇到模型结构或梯度细节问题时再回查阶段 1 与阶段 2</li>
          </ol>
        </div>
      </div>
    </section>

    <!-- ── 六阶段流程图 ───────────────────────────────────────────── -->
    <section class="section">
      <h2>六阶段全景架构 <span class="lead-inline">点击卡片进入对应阶段</span></h2>
      <div class="lead-group">
        <ul class="pts">
          <li><b>阶段 1~2 (模型层):</b> 剖析 Transformer 最小数学闭环与现代主流模型架构组件。</li>
          <li><b>阶段 3~5 (系统层):</b> 依次解决规模化分布式训练、低成本后训练对齐与高吞吐推理服务。</li>
          <li><b>阶段 6 (应用层):</b> 将模型推理封装为具备工具调用、权限控制、状态持久化与子任务隔离的 Agent 系统。</li>
        </ul>
        <p>每张阶段卡片均关联对应子项目的源码目录与章节入口。</p>
      </div>

      <div class="stage-flow">
        <template v-for="(s, i) in stages" :key="s.id">
          <component
            :is="s.status === 'ready' ? 'router-link' : 'div'"
            :to="s.status === 'ready' ? toFor(s) : undefined"
            :class="['stage-card', s.status, `stage-${s.id}`]"
          >
            <div class="stage-head">
              <span class="stage-idx">{{ s.idx }}</span>
              <span class="stage-code mono">{{ s.code }}</span>
              <span v-if="s.status !== 'ready'" :class="['stage-pill', s.status]">待补</span>
            </div>
            <h3 class="stage-title">{{ s.title }}</h3>
            <p class="stage-one">{{ s.oneliner }}</p>
            <!-- 章节名在侧栏里都有, 卡片上只报数 -->
            <div v-if="s.chapters" class="stage-files">
              <span class="mono file-tag">{{ s.groups.length }} 个主题 · {{ s.chapters.length }} 章</span>
            </div>
            <div v-else-if="s.files" class="stage-files">
              <RepoLink
                v-for="f in s.files"
                :key="f"
                :path="`${s.code}${f}`"
                :label="f"
                tiny
              />
            </div>
            <div v-else-if="s.note" class="stage-note">{{ s.note }}</div>
          </component>

          <div v-if="i < stages.length - 1" class="flow-arrow" aria-hidden="true">→</div>
        </template>
      </div>
    </section>

    <!-- ── 阶段 2 时间轴 (llm_models 20 模型) ─────────────────────── -->
    <section class="section">
      <h2>阶段 2 内部 · llm_models 演进时间轴</h2>
      <div class="lead-group">
        <p>时间轴涵盖从 2017 年至 2025 年语言建模、多模态理解与图像视频生成三条技术主线的代表性架构。</p>
        <p>交互方式如下:</p>
        <ol class="pts">
          <li><strong>单击任意模型节点, 可固定展开该模型的组件配置卡片。</strong></li>
          <li>再次点击已选中节点, 或点击卡片底部的「前往本章」, 即可跳转至对应的 PyTorch 源码拆解章节。</li>
        </ol>
      </div>

      <div class="legend">
        <span v-for="(t, k) in tracks" :key="k" class="legend-item">
          <span class="dot" :style="{ background: t.color }" />
          {{ t.label }}
        </span>
      </div>

      <div class="timeline-wrap card">
        <svg class="timeline-svg" :viewBox="`0 0 ${W} ${H}`" width="100%" :height="H">
          <g>
            <line v-for="y in years" :key="'yg-'+y"
                  :x1="xOfYear(y)" :x2="xOfYear(y)"
                  :y1="40" :y2="H - 20"
                  stroke="var(--border)" stroke-dasharray="2 4" />
            <text v-for="y in years" :key="'yt-'+y"
                  :x="xOfYear(y)" :y="H - 6"
                  text-anchor="middle" fill="var(--text-muted)"
                  font-size="11" font-family="SF Mono, Menlo, monospace">{{ y }}</text>
          </g>

          <g>
            <template v-for="(t, k) in tracks" :key="k">
              <line :x1="40" :x2="W - 20" :y1="trackY(k)" :y2="trackY(k)"
                    :stroke="t.color" stroke-opacity="0.25" stroke-width="2" />
              <text :x="4" :y="trackY(k) + 4" fill="var(--text-muted)" font-size="11">
                {{ k === 'left' ? '语言' : k === 'eye' ? '多模态' : '生成' }}
              </text>
            </template>
          </g>

          <!-- 第一次点击 / 回车是「选中」, 卡片留在原地; 对已选中的节点再来一次才跳转。
               悬停和聚焦只是预览。触屏没有悬停, 走的是选中这条路 -->
          <g v-for="m in timeline" :key="m.id"
             @click="select(m.id)" @keydown.enter="select(m.id)" @keydown.space.prevent="select(m.id)"
             @mouseenter="hover = m.id" @mouseleave="hover = null"
             @focus="hover = m.id" @blur="hover = null"
             tabindex="0" role="button" :aria-pressed="pinned === m.id"
             :aria-label="`${m.year} ${m.name}: ${m.blurb}。${pinned === m.id ? '再次按下进入对应章节' : '按回车查看组件配置'}`"
             class="node">
            <circle :cx="nodeX(m)" :cy="trackY(m.track)"
                    :r="shownId === m.id ? 10 : 7"
                    :fill="tracks[m.track].color"
                    stroke="var(--bg-card)" stroke-width="2.5" />
            <text :x="labelX(m)"
                  :y="labelY(m)"
                  text-anchor="middle"
                  font-size="11"
                  :fill="shownId === m.id ? 'var(--text)' : 'var(--text-muted)'"
                  :font-weight="shownId === m.id ? 600 : 400">
              {{ m.name }}
            </text>
          </g>
        </svg>

        <div v-if="hovered" class="hover-card" aria-live="polite">
          <div class="hover-head">
            <span :class="['pill', tracks[hovered.track].cls]">{{ hovered.kind }}</span>
            <strong>{{ hovered.name }}</strong>
            <span class="mono year">{{ hovered.year }}</span>
          </div>
          <p class="desc"><Tex :text="hovered.blurb" /></p>
          <div class="parts">
            <div v-for="(v, k) in hovered.parts" :key="k" class="part">
              <span class="k">{{ labels[k] }}</span>
              <span class="v mono">{{ v }}</span>
            </div>
          </div>
          <div class="hover-foot mono">
            <RepoLink :path="`llm_models/${hovered.file}`" :label="hovered.file" tiny />
            <router-link :to="chapterOf(hovered.id)" class="go">前往本章 →</router-link>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>六个子项目的递进依赖</h2>
      <p class="lead">
        后一阶段均建立在前一阶段的数学或系统抽象之上, 并逐步引入真实生产环境中的算力、显存、延迟与安全约束。
      </p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="path-table">
          <thead>
            <tr>
              <th>阶段</th>
              <th>承接的前序基础</th>
              <th>引入的工程约束</th>
              <th>推荐源码入口</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in pathRows" :key="r.stage">
              <td class="stage-name">{{ r.stage }}</td>
              <td>{{ r.from }}</td>
              <td>{{ r.constraint }}</td>
              <td><RepoLink :path="r.file" tiny /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <!-- 下一章由 ChapterNav 自己从 learningPath 推 -->
    <ChapterNav />
  </div>
</template>

<script setup>
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { tracks, timeline, years, findModel, stages, learningPath } from '@/data/models.js'
import { useProgress } from '@/composables/useProgress.js'
import { LEVELS, inLevel } from '@/data/tiers.js'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import RepoLink from '@/components/RepoLink.vue'
import Tex from '@/components/Tex.vue'

// ── 学习进度 ─────────────────────────────────────────────────────
const progress = useProgress()
const chapters = learningPath.filter((p) => p.route !== 'home')
const resume = computed(() => chapters.find((p) => p.route === progress.state.last) || null)
// 「下一个没读的」只在当前档位里找: 选了冲刺的读者不会被带去扩展章
const nextUnread = computed(() => chapters.find((p) => inLevel(p.route, progress.state.level) && !progress.isVisited(p.route)) || null)
const levelLabel = computed(() => (progress.state.level === 'all' ? '' : `${LEVELS.find((l) => l.id === progress.state.level)?.label || ''}档`))
const coreCount = computed(() => chapters.filter((c) => inLevel(c.route, 'core')).length)
const sprintCount = computed(() => chapters.filter((c) => inLevel(c.route, 'sprint')).length)
// 分母是章节数 (不含序章 home), 和侧栏进度条同一个口径
const readCount = computed(() => chapters.filter((p) => progress.isVisited(p.route)).length)
const masteredCount = computed(() => chapters.filter((p) => progress.isMastered(p.route)).length)

// 时间轴: 悬停是预览, 点击是选中。卡片显示悬停的那个, 没有悬停就显示选中的那个
const router = useRouter()
const hover = ref(null)
const pinned = ref(null)
const select = (id) => (pinned.value === id ? router.push(chapterOf(id)) : (pinned.value = id))
const shownId = computed(() => hover.value || pinned.value)
const hovered = computed(() => shownId.value ? findModel(shownId.value) : null)

const W = 960
const H = 300
const paddingLeft = 70
const paddingRight = 40
const yearMin = Math.min(...years)
const yearMax = Math.max(...years)

const xOfYear = (y) => {
  if (yearMin === yearMax) return W / 2
  return paddingLeft + (W - paddingLeft - paddingRight) * (y - yearMin) / (yearMax - yearMin)
}
const trackY = (k) => ({ left: 80, eye: 160, right: 240 }[k])

const sameSlot = (m) =>
  timeline
    .filter(x => x.year === m.year && x.track === m.track)
    .sort((a, b) => a.id.localeCompare(b.id))

const slotOffset = (m, step) => {
  const same = sameSlot(m)
  if (same.length <= 1) return 0
  const idx = same.findIndex(x => x.id === m.id)
  return (idx - (same.length - 1) / 2) * step
}

const nodeX = (m) => xOfYear(m.year) + slotOffset(m, 30)
const labelX = (m) => xOfYear(m.year) + slotOffset(m, 84)
const labelY = (m) => trackY(m.track) - 28 + slotOffset(m, 18)

const labels = {
  attn: '注意力',
  ffn:  'FFN',
  norm: '归一化',
  pos:  '位置编码',
}

const pathRows = [
  { stage: 'llm_basic',    from: '从 token 输入到交叉熵损失的最小闭环', constraint: '不依赖 autograd, 手写全部反向传播并通过数值梯度校验 (gradcheck)', file: 'llm_basic/model.py' },
  { stage: 'llm_models',   from: '单层单头 Transformer 的基础前向骨架', constraint: '将注意力、FFN、归一化与位置编码模块化, 组合出主流现代架构', file: 'llm_models/layers/core/' },
  { stage: 'llm_train',    from: '单卡前向、反向与参数更新的训练主循环', constraint: '在多卡间切分数据批次、模型层、优化器状态, 并协同混合精度与集合通信', file: 'llm_train/full_loop/demo.py' },
  { stage: 'llm_finetune', from: '预训练基座模型与标准自回归语言模型损失', constraint: '在有限样本与受限显存下完成指令遵循、低秩适配与人类偏好对齐', file: 'llm_finetune/methods/' },
  { stage: 'llm_infer',    from: '微调对齐后的模型权重与自回归解码逻辑', constraint: '同时受首 token 延迟、并发吞吐、KV 显存容量与结构化输出语法约束', file: 'llm_infer/full_engine/' },
  { stage: 'llm_agent',    from: '高吞吐、输出可控的模型推理服务接口', constraint: '围绕模型构建工具执行、权限护栏、上下文压缩、记忆检索与子任务隔离闭环', file: 'llm_agent/full_loop/demo.py' },
]

// 节点 → 章节路由 (集中放在主入口, 避免散落到各章 view 中)
const chapterOf = (id) => {
  const map = {
    transformer: 'blocks', bert: 'blocks',
    gpt3: 'attention', llama: 'attention', mamba: 'models-mamba',
    mistral: 'models-mtp', qwen3_next: 'models-mtp', mtp: 'models-mtp',
    gpt_oss: 'models-gptoss', llada: 'models-llada',
    mixtral: 'moe', deepseek_v3: 'models-moe-balance', deepseek_v32: 'models-dsa',
    clip: 'compare', whisper: 'compare', qwen2_vl: 'position', omni: 'position',
    vae: 'diffusion', dit: 'diffusion', mmdit: 'diffusion',
    video_dit: 'diffusion', vae3d: 'diffusion', var: 'models-var',
  }
  return { name: map[id] || 'compare', query: { focus: id } }
}

// 阶段卡片跳转 — chapters 形式取首章
const toFor = (s) => {
  if (s.route) return { name: s.route }
  if (s.chapters?.length) return { name: s.chapters[0].route }
  return { name: 'home' }
}
</script>

<style scoped>
.route-cta { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; border-left: 3px solid var(--accent); }
.cta-go { display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }
.cta-alt { font-size: 13px; }
/* 导览紧跟副标题, 下面接 ChapterIntro: 上面的间距副标题已经留了, 下面要自己留 */
.onboarding { margin-top: 0; margin-bottom: 28px; }
.cta-k { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.8px; }
.cta-t { font-size: 14px; margin-top: 4px; max-width: 70ch; line-height: 1.7; }
.resume { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; border-left: 3px solid var(--left); }
.resume-k { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.8px; }
.resume-t { font-size: 16px; font-weight: 600; margin: 2px 0 4px; }
.lead-inline {
  font-size: 12px;
  color: var(--text-dim);
  font-weight: 400;
  letter-spacing: 0.2px;
  margin-left: 8px;
}

/* ── 新手导览 + 学习路径 ─────────────────────── */
.onboard-grid, .path-grid { gap: 14px; }
.onboard-card {
  position: relative;
  padding-top: 26px;
}
.onboard-num {
  position: absolute;
  top: -10px;
  left: 16px;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--accent);
  color: #fff;
  display: grid;
  place-items: center;
  font-weight: 600;
  font-size: 13px;
  font-family: "SF Mono", Menlo, monospace;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.onboard-card h3 { margin-bottom: 8px; }
.muted-line { color: var(--text-dim); font-size: 12px; }

.path-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.path-tag {
  align-self: flex-start;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 99px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  font-family: "SF Mono", Menlo, monospace;
  letter-spacing: 0.4px;
}
.path-tag.green  { color: var(--left);  border-color: color-mix(in srgb, var(--left)  40%, var(--border)); }
.path-tag.orange { color: var(--eye);   border-color: color-mix(in srgb, var(--eye)   40%, var(--border)); }
.path-tag.blue   { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 40%, var(--border)); }

.path-card h3 { margin: 4px 0 0; }
.path-steps {
  list-style: none;
  counter-reset: step;
  padding: 0;
  margin: 4px 0 0;
  font-size: 12.5px;
  color: var(--text-muted);
  line-height: 1.7;
}
.path-steps li {
  counter-increment: step;
  position: relative;
  padding-left: 22px;
  padding-bottom: 4px;
}
.path-steps li::before {
  content: counter(step);
  position: absolute;
  left: 0;
  top: 2px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  color: var(--text-dim);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 10px;
  display: grid;
  place-items: center;
}

/* ── 阶段流程卡 ───────────────────────────── */
.stage-flow {
  display: flex;
  align-items: stretch;
  gap: 6px;
  overflow-x: auto;
  padding-bottom: 4px;
}
.stage-card {
  flex: 1 1 0;
  min-width: 220px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-top-width: 3px;
  border-top-color: var(--accent);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  text-decoration: none;
  color: inherit;
  transition: background-color 150ms ease-out, border-color 150ms ease-out, transform 150ms ease-out;
}
.stage-card.ready:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  text-decoration: none;
}
.stage-card.planned {
  border-style: dashed;
  border-top-style: dashed;
  border-top-color: var(--text-dim);
  background: linear-gradient(180deg, var(--bg-elev) 0%, var(--bg-card) 100%);
  cursor: not-allowed;
  opacity: 0.85;
}

.stage-card.stage-basic    { border-top-color: var(--left); }
.stage-card.stage-models   { border-top-color: var(--accent); }
.stage-card.stage-train    { border-top-color: var(--warn); }
.stage-card.stage-finetune { border-top-color: var(--eye); }
.stage-card.stage-infer    { border-top-color: var(--right); }

.stage-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
}
.stage-idx {
  width: 22px; height: 22px;
  border-radius: 50%;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  display: grid; place-items: center;
  font-size: 11px;
  font-family: "SF Mono", Menlo, monospace;
  color: var(--text);
}
.stage-code { color: var(--text-muted); font-size: 11px; }
.stage-pill {
  margin-left: auto;
  font-size: 10px;
  padding: 1px 7px;
  border-radius: 99px;
  font-family: "SF Mono", Menlo, monospace;
  letter-spacing: 0.4px;
}
.stage-pill.planned { background: var(--bg-elev); color: var(--text-dim); border: 1px dashed var(--border); }

.stage-title {
  font-size: 13.5px;
  font-weight: 600;
  margin-bottom: 0;
}
.stage-one {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.55;
  margin: 0;
}
.stage-files {
  display: flex; flex-wrap: wrap; gap: 4px;
  margin-top: auto;
}
.file-tag {
  font-size: 10.5px;
  padding: 2px 6px;
  border-radius: 3px;
  background: var(--code-bg);
  border: 1px solid var(--border);
  color: var(--code-text);
}
.stage-note {
  font-size: 11.5px;
  color: var(--text-dim);
  font-style: italic;
  margin-top: auto;
}

.flow-arrow {
  align-self: center;
  color: var(--text-dim);
  font-size: 18px;
  user-select: none;
  flex-shrink: 0;
}

@media (max-width: 1100px) {
  .stage-flow { flex-direction: column; }
  .flow-arrow { transform: rotate(90deg); }
}

/* ── 时间轴 ───────────────────────────────── */
.legend {
  display: flex;
  gap: 24px;
  padding: 12px 0 16px;
  font-size: 13px;
  color: var(--text-muted);
}
.legend-item { display: inline-flex; align-items: center; gap: 8px; }
.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.timeline-wrap {
  position: relative;
  overflow-x: auto;
}
.timeline-svg {
  display: block;
  min-width: 960px;
}
.node { cursor: pointer; transition: opacity 150ms ease-out; }

.hover-card {
  position: absolute;
  right: 20px;
  top: 20px;
  width: 320px;
  background: var(--bg-elev);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  box-shadow: var(--shadow);
  font-size: 13px;
}
.hover-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.hover-head strong {
  font-size: 15px;
  color: var(--text);
  flex: 1;
}
.hover-head .year {
  color: var(--text-dim);
  font-size: 12px;
}
.hover-card .desc { color: var(--text-muted); margin-bottom: 12px; }
.hover-card .parts { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
.hover-card .part { font-size: 12px; }
.hover-card .part .k { color: var(--text-dim); display: block; font-size: 11px; }
.hover-card .part .v { color: var(--text); }
.hover-card .hover-foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px dashed var(--border);
  font-size: 11px;
  color: var(--text-dim);
}

table.path-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.path-table th {
  text-align: left;
  padding: 12px 14px;
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
  border-bottom: 1px solid var(--border-strong);
}
table.path-table td {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
table.path-table .stage-name { color: var(--text); font-weight: 600; }
table.path-table .small { font-size: 12px; color: var(--text-muted); }
</style>
