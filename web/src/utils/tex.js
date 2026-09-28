// 公式排版: 文案里 $…$ 是行内公式, $$…$$ 是独立成行的公式, \$ 是字面的美元符。
// JS 字符串里反斜杠要写两个 ('$\\sqrt{d}$'), 否则 \s \f \n 会被 JS 先吃掉。npm run check:math 会查。
import katex from 'katex'

const RE = /\$\$((?:\\.|[^$\\])+?)\$\$|\$((?:\\.|[^$\\])+?)\$/g

// 拆成 [{ text } | { tex, display }]
export function texParts(s) {
  const out = []
  let at = 0
  for (const m of String(s ?? '').matchAll(RE)) {
    if (m.index > at) out.push({ text: s.slice(at, m.index) })
    out.push({ tex: m[1] ?? m[2], display: m[1] != null })
    at = m.index + m[0].length
  }
  if (at < String(s ?? '').length) out.push({ text: s.slice(at) })
  return out.map((p) => (p.text != null ? { text: p.text.replaceAll('\\$', '$') } : p))
}

// throwOnError: false —— 写错的公式显示成红字 (.katex-error), 页面不会崩
export const renderTex = (tex, display = false, throwOnError = false) =>
  katex.renderToString(tex, { displayMode: display, throwOnError, strict: 'ignore' })

// 常见笔误: JS 字符串里只写了一个反斜杠, \sqrt 变成 sqrt、\frac 里的 \f 变成换页符。KaTeX 不报错, 只会把命令名当字母排出来。
const BARE = /(?<![\\a-zA-Z])(frac|dfrac|tfrac|sqrt|cdot|times|sigma|theta|alpha|beta|gamma|delta|lambda|epsilon|varepsilon|nabla|approx|leq|geq|neq|odot|otimes|sum|prod|infty|partial|mathbb|mathrm|mathbf|operatorname|text|left|right)(?![a-zA-Z])/
// \text{…} \mathrm{…} 里本来就是普通单词, 不查
const WORDS = /\\(?:text|mathrm|mathbf|mathit|operatorname)\{[^{}]*\}/g
export function lintTex(tex) {
  if (/[\f\v\b\t\r\n]/.test(tex)) return '含控制字符 (反斜杠被 JS 转义吃掉了, 写成 \\\\)'
  const bare = tex.replace(WORDS, '').match(BARE)
  if (bare) return `"${bare[1]}" 前面缺反斜杠`
  try { renderTex(tex, false, true) } catch (e) { return e.message }
  return ''
}

// 搜索用: 把公式源码粗略还原成能搜的文字 (\\lambda → λ, 去掉 $ { } 和其他命令名), 不求排版正确
const GREEK = { alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', Delta: 'Δ', epsilon: 'ε', varepsilon: 'ε', eta: 'η', theta: 'θ',
  kappa: 'κ', lambda: 'λ', mu: 'μ', pi: 'π', rho: 'ρ', sigma: 'σ', Sigma: 'Σ', tau: 'τ', phi: 'φ', psi: 'ψ', Psi: 'Ψ', omega: 'ω', nabla: '∇' }
export const texPlain = (s) =>
  String(s ?? '').replace(/\\([a-zA-Z]+)/g, (_, c) => GREEK[c] ?? '').replace(/[${}\\]/g, '')
