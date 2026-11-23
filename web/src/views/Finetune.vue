<template>
  <div>
    <h1 class="page-title">任务适配 · 从 SFT / LoRA 到 DPO / GRPO / 蒸馏</h1>
    <div class="page-subtitle lead-group">
      <p>
        预训练基座只学会了顺着上文续写, 微调与对齐要解决的是<strong>如何用尽量少的数据、显存和训练链路, 把基座塑造成遵循指令且符合偏好的模型</strong>。
      </p>
      <p>
        <RepoLink path="llm_finetune/" label="llm_finetune/" tiny /> 实现了这一阶段最核心的 16 种算法。本页先立起监督微调 (SFT)、低秩适配 (LoRA) 与直接偏好优化 (DPO) 三根支柱, 后续章节再进一步展开 QLoRA / DoRA、无参考模型偏好对齐、可验证奖励强化学习 (GRPO 家族) 以及知识蒸馏。
      </p>
    </div>

    <ChapterIntro
      tldr="后训练沿着「教格式 → 省显存 → 学偏好」递进: SFT 只对回复区间计算交叉熵, 教会模型按指令作答; LoRA 冻结原模型, 仅训练旁路低秩矩阵就把可训参数压到千分之几; DPO 则利用成对好坏回答直接做对数几率分类, 省去了传统 RLHF 里单独训练奖励模型与跑强化学习的繁琐流程。"
      question="面对同样的偏好数据, 为什么经典 PPO 需要同时维护 4 个模型, 而 DPO 只需 2 个模型配一行 logsigmoid 损失?"
      :goals="[
        '理清 SFT、LoRA 与 DPO 各自所需的数据形态、可训练参数范围与单步前向次数',
        '推导 LoRA 将可训练参数比例压缩至 $2r/d$ 的参数量账本',
        '理解 DPO 如何将奖励建模与 PPO 策略优化合并为单一二分类损失',
      ]"
      :codes="[
        { path: 'llm_finetune/methods/sft.py' },
        { path: 'llm_finetune/methods/lora.py' },
        { path: 'llm_finetune/methods/dpo.py' },
        { path: 'llm_finetune/utils/param_utils.py' },
      ]"
      :prereq="prevChapter"
      :next-step="{ name: 'finetune-sft', label: '下一章 · SFT 数据与 loss' }"
    />

    <StageContents stage-id="finetune" />

    <!-- ── 1. 三阶段 alignment 总览 ───────────────────────────── -->
    <section class="section">
      <h2>1. 三阶段对齐流程与 DPO 的简化</h2>
      <div class="lead-group">
        <p>自 ChatGPT (2022) 确立后训练范式以来, 将通用预训练语言模型转化为指令助手通常包含三个递进阶段:</p>
        <ol class="pts">
          <li><strong>SFT</strong>: 在指令与回复配对数据上做监督微调, 使模型学会按指令格式作答。</li>
          <li><strong>Reward Model (RM)</strong>: 在人类偏好数据上训练奖励模型, 为任意候选回复输出标量打分。</li>
          <li><strong>PPO</strong>: 以强化学习方式在奖励信号引导下更新策略模型, 同时施加 KL 散度约束防止策略偏离过远。</li>
        </ol>
        <p>上述三阶段需要串联三轮独立训练, 其中经典 RLHF 的 PPO 阶段还需在显存中同时常驻 4 个模型:</p>
        <ul class="pts">
          <li><b>经典 RLHF 四模型口径:</b> 包含当前训练的策略模型 (policy)、冻结的参考模型 (ref)、价值网络 (critic) 以及奖励模型 (RM)。</li>
          <li>
            <b>可验证奖励场景的三模型口径:</b> 本教程 <router-link :to="{ name: 'finetune-ppo' }">PPO 章节</router-link>采用 verifier 程序直接判分, 无需加载独立的 RM, 显存中常驻的模型缩减为 policy、ref 与 critic 共 3 份。
          </li>
        </ul>
        <p>DPO (2023) 利用 KL 约束奖励最大化问题的解析解, 将后两个阶段合并为直接在偏好对上计算的二分类损失, 从而省去了独立的 RM 训练与在线 RL 采样。</p>
      </div>

      <EvolutionChain
        title="演进 · 对齐训练链路的逐步收敛"
        subtitle="SFT 建立指令回复能力, RM 拟合人类偏好打分, PPO 在奖励与 KL 约束下更新策略, DPO 则将后两步合并为离线分类损失。"
        :steps="alignSteps"
      />
    </section>

    <!-- ── 2. SFT ──────────────────────────────────────────── -->
    <section class="section">
      <h2>2. SFT · 仅对回复区间计算交叉熵</h2>
      <div class="lead-group">
        <p>
          SFT 与预训练在损失计算上的核心区别在于标签掩码: 将 prompt 对应位置的 labels 设为
          <code class="inline">-100</code>, 交叉熵函数便会在求平均时自动忽略这些位置。
        </p>
        <p><Tex text="实现时最易出错的是错位边界: 由于自回归训练中 labels 已相对输入左移一位, 应当屏蔽的是前 $P-1$ 个位置, 而非前 $P$ 个位置。" /></p>
        <p>
          若多屏蔽一位, 训练损失曲线依然平滑下降, 但模型失去了预测首个回复 token 的监督信号, 无法学会如何开启回答。在本仓库同配置实测中, 这一处越界会使留出集 exact-match 从 1.000 直接降至 0.000。
        </p>
        <p>
          读者可在 <router-link :to="{ name: 'finetune-sft' }">SFT 章节的交互实验台</router-link>中切换掩码边界, 观察逐位置监督信号的变化。
        </p>
      </div>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>labels 掩码构造 <span class="tag">关键边界</span></h3>
          <pre class="code">{{ sftLabels }}</pre>
          <p class="hint">
            屏蔽 prompt 是因为提问文本由外部输入给定, 微调目标是建模条件回复概率而非拟合用户提问分布。
          </p>
          <p class="hint">
            对 labels 施加掩码并不会减少显存占用或前向计算量 (整条拼接序列仍需完整参与前向传播), 其作用是将梯度更新完全集中在 response 区间。
          </p>
        </div>

        <div class="card">
          <h3>SFTLoss 与预训练损失完全一致 <span class="tag">代码对照</span></h3>
          <pre class="code">{{ sftLoss }}</pre>
          <p class="hint">
            本仓库并未单独实现新的损失类: <code class="inline">SFTLoss</code> 只是
            <code class="inline">StandardLMLoss</code> 的直接别名赋值。
          </p>
          <p class="hint">
            SFT 与预训练在目标函数层面 100% 相同, 差异完全体现在数据管道构造的 labels 掩码上。
            对应源码见 <RepoLink path="llm_finetune/methods/sft.py" label="llm_finetune/methods/sft.py" tiny />。
          </p>
        </div>
      </div>
    </section>

    <!-- ── 3. LoRA ─────────────────────────────────────────── -->
    <section class="section">
      <h2>3. LoRA · 在冻结权重旁注入低秩增量</h2>
      <div class="lead-group">
        <p>
          LoRA 原论文 (Hu et al., 2021) 指出: 微调过程中的权重增量 <Tex text="$\Delta W$" /> <strong>本征秩很低</strong>,
          可用一对秩为 <Tex text="$r$" /> 的小矩阵近似表达。训练阶段仅更新这对旁路矩阵, 推理阶段将其合并回主干权重即可实现零额外延迟。
        </p>
        <ul class="pts">
          <li><b>核心收益:</b> 显著压缩训练显存与落盘体积, 支持多租户场景下按请求快速切换 adapter。</li>
          <li><b>能力边界:</b> 单步收敛速度不及全参微调。在本仓库同基座各训练 300 步的对照实验中, 全参微调留出集 EM 为 0.809, 而 <Tex text="LoRA ($r=8$)" /> 为 0.352。</li>
        </ul>
      </div>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>数学形式 <span class="tag">3 行核心定义</span></h3>
          <Prose :text="loraMath" />
          <p class="hint">
            <strong>零初始化保起点</strong>: 令 <Tex text="$B = 0$" />, 则第 0 步时 <Tex text="$BA = 0$" />, 前向输出与原始预训练模型逐位一致, 保证微调从预训练终态平滑起步而不会在初始步破坏已有表征。
          </p>
          <p class="hint">将 A 或 B 任一置零均可满足初始 <Tex text="$BA = 0$" />, 但梯度行为不同:</p>
          <ul class="hint pts">
            <li><b>仅将其中一个置零:</b> 被置零的矩阵在第 1 步即可获得非零梯度, 另一矩阵从第 2 步起开始更新。</li>
            <li><b>将两个矩阵同时置零:</b> 双方梯度恒为 0, 参数处于鞍点而无法启动训练。</li>
          </ul>
        </div>

        <div class="card">
          <h3>参数量账本 · 为何能省 99.6% <span class="tag">定量对比</span></h3>
          <table class="cmp">
            <thead>
              <tr>
                <th>方案</th>
                <th class="num">可训练参数</th>
                <th>注释</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>全参微调</td>
                <td class="num mono"><Tex text="$d_{\text{in}} \times d_{\text{out}}$" /></td>
                <td class="muted"><Tex text="$4096^2 \approx 16\text{M}$" /> (单层注意力投影矩阵)</td>
              </tr>
              <tr class="hl">
                <td><Tex text="LoRA ($r=8$)" /></td>
                <td class="num mono"><Tex text="$r \times (d_{\text{in}} + d_{\text{out}})$" /></td>
                <td class="muted"><Tex text="$8 \times 8192 \approx 65\text{K}$" /> (约为原参数的 0.4%)</td>
              </tr>
            </tbody>
          </table>
          <ul class="hint pts">
            <li><b>秩 <Tex text="$r$" />:</b> 目标任务越复杂、有效训练样本越多, 所需的秩 <Tex text="$r$" /> 通常越大。</li>
            <li><b>缩放因子 <Tex text="$\alpha$" />:</b> 常取 <Tex text="$2r$" /> 或 <Tex text="$r$" />。乘以缩放比 <Tex text="$\alpha/r$" /> 后, 调整 <Tex text="$r$" /> 时输出幅度保持稳定, 无需重新搜索学习率。</li>
            <li><b>学习率 lr:</b> LoRA 的学习率通常需设为全参微调的 3~10 倍, 因为矩阵 B 从零起步, 早期 <Tex text="$\Delta W$" /> 的有效更新步幅较小。</li>
          </ul>
        </div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h3>LoRALinear · 训练双分支与推理合并路径</h3>
        <DagView v-bind="loraDag" />
        <ul class="hint pts">
          <li><b>推理合并 <code class="inline">merge()</code>:</b> 执行 <Tex text="$W \leftarrow W + (\alpha/r)\cdot BA$" />, 此后前向直接跳过旁路分支。</li>
          <li><b>恢复训练 <code class="inline">unmerge()</code>:</b> 执行 <Tex text="$W \leftarrow W - (\alpha/r)\cdot BA$" />, 还原主干加旁路的双分支计算图。</li>
        </ul>
        <div class="lora-points">
          <div class="point">
            <strong>训练阶段</strong>
            <span class="muted">前向传播同时经过主干与旁路分支, 主干权重的 requires_grad 设为 False, 反向传播仅计算并更新 A 与 B 的梯度。</span>
          </div>
          <div class="point">
            <strong>推理阶段</strong>
            <span class="muted">调用 <code class="inline">merge()</code> 将旁路增量 <Tex text="$(\alpha/r)\cdot BA$" />
            并入 base.weight 后, 前向传播退化为单分支投影, 推理延迟与原始 nn.Linear 完全一致。</span>
          </div>
          <div class="point">
            <strong>仅落盘适配器</strong>
            <span class="muted"><code class="inline">get_lora_state_dict</code> 仅提取 A 与 B 两个矩阵:
            本仓库配置下适配器占 76 KiB, 完整模型权重占 389 KiB, 可共享同一份基座按需分发适配器。</span>
          </div>
        </div>
        <p class="hint">
          对应源码: <RepoLink path="llm_finetune/methods/lora.py:LoRALinear" label="llm_finetune/methods/lora.py:LoRALinear / apply_lora / merge_lora_weights / get_lora_state_dict" tiny />
        </p>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h3>目标注入层 · target_modules <span class="tag">LLaMA 实践经验</span></h3>
        <table class="cmp">
          <thead>
            <tr><th>策略</th><th>注入模块</th><th>工程取舍</th></tr>
          </thead>
          <tbody>
            <tr><td>最小化注入</td><td class="mono">w_q, w_v</td><td class="muted">原论文基础配置, 单位可训练参数收益最高</td></tr>
            <tr class="hl"><td>推荐配置 (本仓库默认)</td><td class="mono">w_q, w_k, w_v, w_o</td><td class="muted">覆盖注意力层全部投影矩阵, 效果与参数开销较均衡</td></tr>
            <tr><td>全线性层注入</td><td class="mono">attn 全部 + ffn 全部</td><td class="muted">QLoRA 论文采用的配置, 拟合上限最高但 adapter 参数量约翻倍</td></tr>
          </tbody>
        </table>
        <p class="hint">
          <code class="inline">apply_lora</code> 遍历 <code class="inline">named_modules</code>
          并匹配模块路径的末尾属性名 (例如「w_q」), 不耦合具体的层号或父级容器路径,
          因此同一套注入代码可直接复用于 LLaMA、Mistral 等沿用相同投影命名的模型。
        </p>
        <p class="hint">LoRA 的两个重要演进变体将在后续专章展开:</p>
        <ul class="hint pts">
          <li>
            <router-link :to="{ name: 'finetune-qlora' }">QLoRA</router-link>:
            将冻结的基座权重进一步量化至 4 bit (采用 NF4 分位数码本配合每 block 独立的 absmax 缩放因子, 平均约 4.5 bit/参数)。
            本仓库实测整模型占用从 389 KiB 降至 59 KiB, 对应实现见
            <RepoLink path="llm_finetune/methods/qlora.py" label="llm_finetune/methods/qlora.py" tiny />。
          </li>
          <li>
            <router-link :to="{ name: 'finetune-dora' }">DoRA</router-link>:
            将权重矩阵分解为「幅度与方向」两部分, 令低秩旁路专注于调整方向分量, 同时单独学习幅度标量向量。
          </li>
        </ul>
      </div>
    </section>

    <!-- ── 4. DPO ──────────────────────────────────────────── -->
    <section class="section">
      <h2>4. DPO · 将偏好学习化简为一行 logsigmoid 损失</h2>
      <div class="lead-group">
        <p>
          DPO 利用了带 KL 散度约束的奖励最大化问题存在<strong>解析最优解</strong>这一性质。将该解析关系代回成对比较的 Bradley-Terry 偏好模型后,
          优化目标直接化简为衡量「正样本相对负样本的对数概率比优势」的二分类损失, 从而省去了独立的 RM 训练与 PPO 在线强化学习。
        </p>
        <p>
          相应代价在于该目标仅约束正负样本的相对差值, 因此训练过程中
          <router-link :to="{ name: 'finetune-dpo' }">chosen 自身的绝对对数概率可能同步下降</router-link>
          (本仓库实测 <Tex text="$\log \pi(\text{chosen})$" /> 从 −4.03 降至 −4.18, 贪心解码 EM 从 0.332 降至 0.137)。
        </p>
      </div>

      <div class="grid grid-2" style="gap: 16px;">
        <div class="card">
          <h3>核心公式 <span class="tag">Bradley-Terry</span></h3>
          <Prose :text="dpoMath" />
          <ul class="hint pts">
            <li><b><Tex text="$\pi_\theta$" />:</b> 待更新的策略模型 (policy), 通常以 SFT 终态权重作为微调起点。</li>
            <li><b><Tex text="$\pi_{\text{ref}}$" />:</b> 冻结的参考模型 (reference), 通常直接拷贝自同一份 SFT 终态权重。</li>
            <li><b><Tex text="$\beta$" />:</b> KL 约束强度系数, 常取 0.1~0.5。<Tex text="$\beta$ 越大, $\sigma$ 函数越快进入饱和区" />, 策略越严格贴近参考模型, 因此它控制的是偏离惩罚强度而非更新步幅。</li>
          </ul>
          <p class="hint">
            在第 0 步时策略模型与参考模型权重完全相同, 两项对数概率比均为 0, 初始损失精确等于 <Tex text="$-\log\sigma(0) = \ln 2 \approx 0.6931$" />。
            若初始损失偏离该数值, 通常说明参考模型权重未对齐或 labels 掩码存在越界。
          </p>
        </div>

        <div class="card">
          <h3>单步训练流程 <span class="tag">2 次 LM 前向</span></h3>
          <pre class="code">{{ dpoStep }}</pre>
          <p class="hint">
            实现时将 chosen 与 rejected 沿批次维拼接为 2B 条序列的 batch, 由策略模型与参考模型各执行一次前向传播: 单步共计 2 次 LM 前向,
            显存中常驻两份模型权重 (本仓库配置下共 778 KiB)。
          </p>
          <p class="hint">参考模型前向无需计算梯度也不保存中间激活值, 因此实测单步耗时仅增加约 35%, 而非翻倍增加 100%。</p>
          <ul class="hint pts">
            <li><b><code class="inline">no_grad</code>:</b> 免去中间激活缓存与反向图构建, 节省激活显存。</li>
            <li><b><code class="inline">eval()</code>:</b> 关闭 dropout 等随机算子, 确保 <Tex text="$\log \pi_{\text{ref}}$" /> 为确定性基准函数, 避免随机噪声污染偏好梯度。</li>
          </ul>
          <p class="hint">
            对应源码: <RepoLink path="llm_finetune/methods/dpo.py:DPOLoss" label="llm_finetune/methods/dpo.py:DPOLoss / PairwiseForward" tiny />
          </p>
        </div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h3>compute_sequence_logprobs · 序列级对数概率计算</h3>
        <pre class="code">{{ seqLogprob }}</pre>
        <p class="hint">
          此处不能直接调用 <code class="inline">F.cross_entropy(reduction='sum')</code>:
          该接口会将整个 batch 内全部样本的负对数似然累加为单一标量, 从而丢失逐样本边界。
          DPO 必须先算出<strong>每条样本各自的序列对数概率和 <Tex text="$\sum \log p$" /></strong>, 才能按配对计算 chosen 与 rejected 的差值。
        </p>
        <p class="hint">
          有效求和区间必须包含<strong>首个回复 token</strong>。若 prompt 掩码多屏蔽一位, 便会漏掉首 token 条件对数概率 <Tex text="$\log p(y_1 \mid x)$" />,
          而正负样本在首个回复 token <Tex text="$y_1$" /> 上往往已经出现分歧。
        </p>
      </div>
    </section>

    <!-- ── 5. 三柱对照 ─────────────────────────────────────── -->
    <section class="section">
      <h2>5. 三大支柱横向对照</h2>
      <p class="lead">围绕同一基座模型, 下表从数据形态、可训练参数、前向开销与落盘产物八个维度对比三种核心后训练方法。</p>
      <div class="card" style="padding: 0; overflow-x: auto;">
        <table class="three">
          <thead>
            <tr>
              <th>维度</th>
              <th class="col-sft">SFT</th>
              <th class="col-lora">LoRA</th>
              <th class="col-dpo">DPO</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in pillars" :key="row.dim">
              <td class="dim">{{ row.dim }}</td>
              <td class="mono"><Tex :text="row.sft" /></td>
              <td class="mono"><Tex :text="row.lora" /></td>
              <td class="mono"><Tex :text="row.dpo" /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ── 6. 决策树 ─────────────────────────────────────────── -->
    <section class="section">
      <h2>6. 算法选型参考</h2>
      <div class="card">
        <ol class="decision">
          <li v-for="d in decisions" :key="d.q">
            <span class="q">{{ d.q }}</span>
            <span class="a">→ {{ d.a }}</span>
          </li>
        </ol>
        <p class="hint">工程实践中常见的一条串联路线是从基座模型出发, 依次执行 <strong>LoRA-SFT、合并或解锁基座、DPO 偏好对齐</strong>:</p>
        <ol class="hint pts">
          <li>先以低显存开销的 LoRA 在指令数据上完成格式预热。</li>
          <li>再基于预热后的策略模型, 在成对偏好数据上执行 DPO 对齐。</li>
        </ol>
        <p class="hint">
          上表仅提供方法选型的定性方向。若需定量对比单步前向次数、常驻权重份数与落盘体积,
          可前往<router-link :to="{ name: 'finetune-runs' }">训练脚本与落盘章节的选型计算器</router-link>。
        </p>
      </div>
    </section>

    <!-- 本章挂载的实验台 (data/labmap/*.js) 与章末自测 (data/quiz/*.js), 没配置时不渲染 -->

    <LabMount />

    <QuizCard />


    <ChapterNav
      :prev="{ ...prevChapter, hint: '回顾分布式训练系统如何组装完整主循环' }"
      :next="{ name: 'finetune-sft', label: '下一章 · SFT 数据与 loss', hint: '从数据打包与 labels 掩码边界开始拆解微调' }"
    />
  </div>
</template>

<script setup>
import StageContents from '@/components/StageContents.vue'
import LabMount from '@/components/LabMount.vue'
import QuizCard from '@/components/QuizCard.vue'
import ChapterIntro from '@/components/ChapterIntro.vue'
import ChapterNav from '@/components/ChapterNav.vue'
import EvolutionChain from '@/components/EvolutionChain.vue'
import RepoLink from '@/components/RepoLink.vue'
import Prose from '@/components/Prose.vue'
import Tex from '@/components/Tex.vue'
import DagView from '@/components/dag/DagView.vue'
import { learningPath } from '@/data/models.js'

// 上一章从 learningPath 取, 不手写编号 (训练阶段加章后手写的 "3.5" 会过期; 与 Infer.vue 同一写法)
const prevItem = learningPath[learningPath.findIndex((x) => x.route === 'finetune') - 1]
const prevChapter = { name: prevItem.route, label: `上一章 · ${prevItem.label}` }

const alignSteps = [
  { name: 'SFT',        year: '2022',
    pain: '预训练基座仅拟合无条件文本续写分布, 不会遵循用户指令作答。',
    fix:  '在 (instruction, response) 配对样本上计算自回归 CE 损失, 并将 prompt 区间标签置为 -100。',
    color: 'var(--left)' },
  { name: 'Reward Model', year: '2022',
    pain: 'SFT 只能模仿单一参考回复, 无法刻画同一指令下不同候选回答的优劣偏好。',
    fix:  '采集 (prompt, chosen, rejected) 成对偏好数据, 训练标量奖励模型为任意候选回复打分。',
    color: 'var(--eye)' },
  { name: 'PPO',        year: '2022',
    pain: '奖励模型可评估回复质量但不可直接对离散采样求导, 且直接最大化奖励易引发策略退化。',
    fix:  '以 PPO 算法在奖励信号与相对 ref 的 KL 惩罚下在线更新策略, 经典配置需同时维护 4 个模型。',
    color: 'var(--right)' },
  { name: 'DPO',        year: '2023',
    pain: 'PPO 训练链路繁琐、显存开销高且超参敏感, 奖励模型还需单独训练与校验。',
    fix:  '导出解析损失 $-\\log\\sigma(\\beta\\cdot(\\log\\pi_\\theta - \\log\\pi_{\\text{ref}})\\text{ 差值})$, 直接在离线偏好对上做二分类优化, 省去 RM 与在线 RL。',
    color: 'var(--accent)' },
]

const sftLabels = `# 例: x = [<bos>] [Q1] [Q2] [Q3] [A1] [A2] [<eos>], prompt 长 P = 4
idx    = x[:-1] = [<bos>] [Q1]  [Q2]  [Q3] [A1] [A2]
labels = x[1:]  = [-100 ] [-100][-100][A1] [A2] [<eos>]
                   ↑ 只 mask 前 P-1 = 3 格  ↑ 看到 Q3 要预测 A1: 这一格必须留

# pad 也置 -100`

const sftLoss = `# llm_finetune/methods/sft.py 的全部内容 (去掉 docstring):
from llm_models.training.loss import StandardLMLoss

SFTLoss = StandardLMLoss        # ← 别名, 没有新实现

# StandardLMLoss 里就是一句:
#   F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
#                   ignore_index=-100)     # prompt / pad 在这里被跳过`

const loraMath = `冻结主干线性层:
$$y = Wx, \\quad W \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$$
注入低秩旁路后:
$$\\begin{aligned} y &= Wx + (\\alpha/r)\\cdot BAx \\\\ \\text{其中 } A &\\in \\mathbb{R}^{r \\times d_{\\text{in}}},\; B \\in \\mathbb{R}^{d_{\\text{out}} \\times r},\; r \\ll \\min(d_{\\text{in}}, d_{\\text{out}}) \\end{aligned}$$
参数初始化策略:
- $A$ 采用 Kaiming 正态分布初始化, 与 nn.Linear 默认初始化对齐。
- $B$ 初始化为全零矩阵, 使第 0 步旁路增量 $BA = 0$, 保证微调起点与预训练终态完全一致。`

// x 分两路: 冻结的 base 一路直通, A/B 旁路一路, 在 y 处相加。
// 旁路可以整个折起来, 折起来就是 merge() 之后的样子。
const loraDag = {
  dir: 'LR',
  nodes: [
    { id: 'x', label: 'x', kind: 'io' },
    { id: 'base', label: 'base (W, b)', sub: '[冻结]', kind: 'trunk', note: 'requires_grad = False' },
    { id: 'drop', label: 'dropout', group: 'lora' },
    { id: 'a', label: 'A^T', sub: 'D → r', kind: 'attn', group: 'lora', note: 'Kaiming 正态初始化' },
    { id: 'b', label: 'B^T', sub: 'r → D', kind: 'ffn', group: 'lora', note: '零矩阵初始化, 起点与预训练终态一致' },
    { id: 'y', label: 'y', kind: 'io' },
  ],
  edges: [
    { from: 'x', to: 'base', kind: 'main' },
    { from: 'base', to: 'y', kind: 'main' },
    { from: 'x', to: 'drop' }, { from: 'drop', to: 'a' }, { from: 'a', to: 'b' },
    { from: 'b', to: 'y', label: '× (α/r)' },
  ],
  groups: [{ id: 'lora', label: 'LoRA 旁路 (仅此分支计算梯度)' }],
}

const dpoMath = `$$\\begin{aligned} \\mathcal{L}_{\\text{DPO}} = -\\,\\mathbb{E}_{(x, y_w, y_l)} \\log \\sigma\\Big( \\beta \\cdot \\Big[ &\\log \\frac{\\pi_\\theta(y_w \\mid x)}{\\pi_{\\text{ref}}(y_w \\mid x)} \\\\ &- \\log \\frac{\\pi_\\theta(y_l \\mid x)}{\\pi_{\\text{ref}}(y_l \\mid x)} \\Big] \\Big) \\end{aligned}$$
公式含义:
- 拉高正样本 $y_w$ (chosen) 的序列对数概率, 同时压低负样本 $y_l$ (rejected) 的序列对数概率。
- 两项对数概率均以参考策略 $\\pi_{\\text{ref}}$ 为基准取比值, 隐式施加 KL 散度约束以抑制策略漂移。
- Sigmoid 函数 $\\sigma$ 将无界的隐式奖励差映射至 $(0, 1)$ 区间, 在偏好对上构成稳定的二分类交叉熵目标。`

const dpoStep = `# PairwiseForward.with_frozen_copy(policy) 包住的一步
# chosen 与 rejected 拼成一个 2B 条的 batch, 一次前向算完
both = cat([chosen.input_ids, rejected.input_ids])   # [2B, T]

p_logits = model(both)                 # 1) policy 前向, 要梯度
with torch.no_grad():
    r_logits = ref(both)               # 2) ref 前向, 不要梯度、不存激活

loss = DPOLoss(beta).compute(
    {"policy_chosen_logits": ..., "ref_chosen_logits": ..., ...},
    {"chosen_labels": ..., "rejected_labels": ...},
)["total_loss"]
loss.backward();  optimizer.step()     # 没有 DPOTrainer, 走通用 Trainer`

const seqLogprob = `def compute_sequence_logprobs(logits, labels, ignore=-100):
    log_p_full = F.log_softmax(logits, dim=-1)            # [B, T, V]
    valid      = labels != ignore                         # [B, T]

    safe_labels = labels.masked_fill(~valid, 0)            # 让 gather 合法
    token_logp  = log_p_full.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    return (token_logp * valid.float()).sum(dim=-1)        # [B]  ← 逐样本求和`

const pillars = [
  { dim: '数据形态',     sft: '(instruction, response)',  lora: '(instruction, response)',  dpo: '(prompt, chosen, rejected)' },
  { dim: '可训练参数',   sft: '100% 模型参数',            lora: '约为 $2r/d$ ($d=4096, r=8$ 时为 0.39%)', dpo: '100% policy 参数, ref 冻结' },
  { dim: '损失函数',     sft: '带 -100 掩码的 CE',        lora: '与 SFT 相同 (仅更新旁路参数)', dpo: '$-\\log\\sigma(\\beta\\cdot \\text{logit\\_diff})$' },
  { dim: '是否需 ref',   sft: '否',                       lora: '否',                       dpo: '是 (deepcopy + freeze + eval)' },
  { dim: '是否需 RM',    sft: '否',                       lora: '否',                       dpo: '否 (隐式奖励替代独立 RM)' },
  { dim: '单步 LM 前向', sft: '1 次',                     lora: '1 次',                     dpo: '2 次 (chosen 与 rejected 拼为 2B 条, policy 与 ref 各 1 次)' },
  { dim: '常驻权重',     sft: '1 份 (389 KiB)',           lora: '1 份冻结基座 + 适配器',     dpo: '2 份 (778 KiB)' },
  { dim: '落盘产物',     sft: '1 份完整权重',             lora: '适配器 (本仓库 76 KiB) + 共享基座', dpo: '1 份完整权重' },
]

const decisions = [
  { q: '仅有指令问答对, 且显存充裕?',                 a: '全参 SFT: 拟合上限最高, 相同步数下收敛更快' },
  { q: '仅有指令问答对, 但显存受限?',                 a: 'LoRA SFT: 可训练参数约为 2r/d, 多任务 adapter 易于热切换' },
  { q: '冻结的半精度基座也无法放入单卡显存?',          a: 'QLoRA: 基座采用 NF4 量化 (约 4.5 bit/参数), 仅训练低秩旁路' },
  { q: '固定秩 r 不变, 希望进一步逼近全参微调效果?',   a: 'DoRA: 低秩旁路仅更新方向分量, 幅度向量独立学习' },
  { q: '已有 SFT 模型, 并收集到成对偏好数据?',         a: 'DPO: 省去独立 RM 与在线 PPO, 单一分类损失完成对齐' },
  { q: '有成对偏好数据, 但希望省去参考模型显存?',      a: 'SimPO / ORPO: 单步前向次数与常驻权重份数均减半' },
  { q: '希望将指令微调与偏好对齐合并为单阶段?',        a: 'ORPO: 损失函数内置 NLL 监督项与优势比偏好项' },
  { q: '任务结果可由程序自动验真 (数学推理 / 代码)?',  a: 'GRPO 家族 (DAPO · Dr.GRPO · GSPO): 在线采样配合可验证奖励' },
  { q: '拥有强教师模型, 需将能力迁移至轻量学生模型?',  a: '先做离线蒸馏建立基础能力, 再用 on-policy 蒸馏对齐分布' },
]
</script>

<style scoped>
table.cmp,
table.three {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.cmp th, table.cmp td,
table.three th, table.three td {
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: middle;
}
table.cmp th, table.three th {
  background: var(--bg-elev);
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.7px;
}
table.cmp tr.hl, table.three tr:hover { background: var(--bg-elev); }
table.cmp .num { text-align: right; }
table.cmp .muted, table.three .muted { color: var(--text-dim); }

table.three th.col-sft  { color: var(--left); }
table.three th.col-lora { color: var(--accent); }
table.three th.col-dpo  { color: var(--eye); }
table.three .dim { color: var(--text-muted); width: 130px; }

.lora-points {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 14px;
}
@media (max-width: 960px) {
  .lora-points { grid-template-columns: 1fr; }
}
.point {
  padding: 10px 12px;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.point strong { font-size: 12px; color: var(--text); }
.point .muted { color: var(--text-muted); font-size: 12px; line-height: 1.55; }

.hint {
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
  line-height: 1.6;
}

.decision {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.decision li {
  display: flex;
  gap: 12px;
  padding: 10px 12px;
  background: var(--bg-elev);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--accent);
  font-size: 13px;
}
.decision .q { color: var(--text); flex: 1; }
.decision .a { color: var(--accent); font-family: "SF Mono", Menlo, monospace; font-size: 12.5px; }
</style>
