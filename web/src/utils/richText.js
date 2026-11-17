// 正文排版的「视觉锚点」: 不改文字, 只给读者的眼睛分层。
//   - 括号注解 ` (…中文…)`: 变浅、略小, 懂的人一眼跳过, 不懂的人往下读就有解释
//   - 代码标识符 (带 _ 或 . 或以 () 结尾): 等宽 + 浅底, 和正文分开
//   - 关键数字 (两位以上、小数、科学计数、带 % 或 ×): 加粗, 数据一眼可见
// 只处理纯文字片段 (公式已经被 utils/tex.js 的 texParts 切走)。
// 返回 [{ text, kind?: 'gloss' | 'code' | 'num', parts? }], gloss 的 parts 里还会有 code / num。

// 半角写法 ` (…)` (数据原文) 和中文括号 `（…）` (经过 zhPunct 之后) 都认
const GLOSS = / \(([^()]*[一-鿿][^()]*)\)|（([^（）]*[一-鿿][^（）]*)）/g
const TOKEN = /(?<![A-Za-z0-9_./])([A-Za-z_][A-Za-z0-9_]*(?:[./][A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?)|(?<![A-Za-z0-9_.])(\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?(?:%|×)?)/g

const isCode = (s) => /[_.]/.test(s) || s.endsWith('()')
const isKeyNum = (s) => /[.e%×,]/.test(s) || s.replace(/\D/g, '').length >= 2

function tokens(s) {
  const out = []
  let at = 0
  const push = (text, kind) => {
    if (!text) return
    const last = out[out.length - 1]
    if (!kind && last && !last.kind) last.text += text
    else out.push(kind ? { text, kind } : { text })
  }
  for (const m of s.matchAll(TOKEN)) {
    push(s.slice(at, m.index))
    const [w, ident, num] = m
    push(w, ident ? (isCode(ident) ? 'code' : null) : isKeyNum(num) ? 'num' : null)
    at = m.index + w.length
  }
  push(s.slice(at))
  return out
}

export function richTokens(s) {
  const out = []
  let at = 0
  for (const m of String(s ?? '').matchAll(GLOSS)) {
    const lead = m[0].startsWith(' ') ? 1 : 0 // 半角写法括号前的空格留在正文里
    out.push(...tokens(s.slice(at, m.index + lead)))
    out.push({ kind: 'gloss', text: m[0].slice(lead), parts: tokens(m[0].slice(lead)) })
    at = m.index + m[0].length
  }
  out.push(...tokens(String(s ?? '').slice(at)))
  return out
}

// 一句话的边界: 第一个句号 / 问号 / 感叹号 (公式 $…$ 里的不算)。用来把要点的第一句 (结论) 单独加粗。
export function splitLead(s) {
  let inTex = false
  for (let i = 0; i < s.length; i++) {
    const c = s[i]
    if (c === '\\') { i++; continue }
    if (c === '$') inTex = !inTex
    else if (!inTex && '。?!'.includes(c)) return [s.slice(0, i + 1), s.slice(i + 1)]
  }
  return [s, '']
}
