<!--
  一段可能夹着公式的文字: $…$ 行内, $$…$$ 独立成行。语法见 utils/tex.js。
  只有公式部分走 v-html (KaTeX 的输出), 普通文字照旧转义。
  rich: 正文里的括号注解变浅、代码标识符等宽、关键数字加粗 (utils/richText.js), 不改文字本身。
  文字片段一律先过 zhPunct (utils/zhPunct.js): 数据里写半角, 页面上显示中文标点; 公式和代码不转。
-->
<template>
  <template v-for="(p, i) in parts" :key="i">
    <span v-if="p.html" :class="p.display ? 'tex-block' : 'tex'" v-html="p.html" />
    <template v-else-if="p.rich">
      <template v-for="(t, j) in p.rich" :key="j">
        <span v-if="t.kind === 'gloss'" class="rt-gloss"><template v-for="(u, k) in t.parts" :key="k"><code v-if="u.kind === 'code'" class="rt-code">{{ u.text }}</code><b v-else-if="u.kind === 'num'" class="rt-num">{{ u.text }}</b><template v-else>{{ u.text }}</template></template></span>
        <code v-else-if="t.kind === 'code'" class="rt-code">{{ t.text }}</code>
        <b v-else-if="t.kind === 'num'" class="rt-num">{{ t.text }}</b>
        <template v-else>{{ t.text }}</template>
      </template>
    </template>
    <template v-else>{{ p.text }}</template>
  </template>
</template>

<script setup>
import { computed } from 'vue'
import { renderTex, texParts } from '@/utils/tex'
import { richTokens } from '@/utils/richText'
import { zhPunct } from '@/utils/zhPunct'

const props = defineProps({ text: { type: [String, Number], default: '' }, rich: { type: Boolean, default: false } })
const parts = computed(() => texParts(String(props.text)).map((p) =>
  p.tex != null ? { html: renderTex(p.tex, p.display), display: p.display }
    : props.rich ? { rich: richTokens(zhPunct(p.text)) } : { text: zhPunct(p.text) }))
</script>
