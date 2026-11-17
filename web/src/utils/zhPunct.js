// 显示层的中文标点: 数据和 README 里照旧写半角 (`, ` `: ` `; ` `?` `( )`), 页面上统一显示成中文标点 (，：；？（）)。
// 只转换「正文」标点, 不碰代码:
//   - 紧跟在标识符后面的 ( 是函数调用 / 下标, 连同里面的内容原样保留: next(messages, tool_schemas)
//   - [ ] 和 { } 里面原样保留: pool[table[pos // bs], pos % bs]、{type: tool_use, id}
//   - 冒号、逗号、分号后面必须跟空格 (或在句尾) 才转: 1:9、12:30、https://、1,000 都不动
//   - 没有汉字的字符串整段不动 (纯英文、纯代码、数字)
// 公式由调用方先切走 (utils/tex.js 的 texParts), 这里只处理普通文字。
// 同一个函数在三处用: Tex.vue 的文字片段、模板静态文字 (vite.config.js 编译期)、模板插值 (全局 $zh)。

const HAN = /[㐀-鿿]/
const ID_TAIL = /[A-Za-z0-9_\]]/
const MAP = { ',': '，', ';': '；', ':': '：' }
const AFTER_Q = /[㐀-鿿」）』"'”’]/

export function zhPunct(s) {
  if (typeof s !== 'string' || !HAN.test(s)) return s
  let out = ''
  const stack = [] // 每层括号是 'code' 还是 'prose'
  const inCode = () => stack.includes('code')
  for (let i = 0; i < s.length; i++) {
    const c = s[i]
    const next = s[i + 1]
    if (c === '(') {
      const code = inCode() || (i > 0 && ID_TAIL.test(s[i - 1]))
      stack.push(code ? 'code' : 'prose')
      if (code) out += c
      else out = out.replace(/ $/, '') + '（' // 「词 (」→「词（」
      continue
    }
    if (c === ')') {
      const top = stack.pop()
      if (top === 'code') { out += c; continue }
      out += '）' // 正文括号, 或者左括号在公式另一侧 (跨片段) 的右括号
      if (next === ' ') i++
      continue
    }
    if (c === '[' || c === '{') { stack.push('code'); out += c; continue }
    if (c === ']' || c === '}') { if (stack.at(-1) === 'code') stack.pop(); out += c; continue }
    if (!inCode()) {
      if (c in MAP && (next === ' ' || next === undefined)) {
        out += MAP[c]
        if (next === ' ') i++
        continue
      }
      if ((c === '?' || c === '!') && (next === undefined || next === ' ' || AFTER_Q.test(next))) {
        out += c === '?' ? '？' : '！'
        if (next === ' ') i++
        continue
      }
    }
    out += c
  }
  return out
}
