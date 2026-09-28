// 校验全站公式 ($…$ / $$…$$): KaTeX 能解析, 且没有"反斜杠被 JS 吃掉"的笔误。
// KaTeX 遇到 sqrt{d} (少了 \) 不会报错, 只会排成字母, 页面上静默出错, 所以放进 CI。  用法: npm run check:math
import { readdirSync, statSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, join, relative } from 'node:path'
import { lintTex, texParts } from '../src/utils/tex.js'

const src = join(dirname(fileURLToPath(import.meta.url)), '../src')
const walk = (d) => readdirSync(d).flatMap((f) => (statSync(join(d, f)).isDirectory() ? walk(join(d, f)) : [join(d, f)]))
const files = walk(src)
let n = 0
const bad = []
const lint = (tex, where) => { n++; const e = lintTex(tex); if (e) bad.push(`✗ ${where}: $${tex}$  → ${e}`) }

// 1. data/**/*.js: 直接 import, 查运行时的真实字符串。
// 用到 import.meta.glob 的汇总文件 (直接或间接) 在 node 里 import 不了, 跳过; 它们的内容都来自下面那些分文件
const skipped = []
const strings = (v, path, out) => {
  if (typeof v === 'string') out.push([v, path])
  else if (v && typeof v === 'object') for (const [k, x] of Object.entries(v)) strings(x, `${path}.${k}`, out)
  return out
}
for (const f of files.filter((f) => f.includes('/data/') && f.endsWith('.js'))) {
  let mod
  try { mod = await import(pathToFileURL(f)) } catch { skipped.push(relative(src, f)); continue }
  for (const [s, path] of strings(mod, relative(src, f), []))
    for (const p of texParts(s)) if (p.tex != null) lint(p.tex, path)
}

// .vue 里的公式这里不查: 源码里分不清公式的 $ 和代码里的 $ (${} 插值、$event、正则), 静态扫描全是误报。
// 它们靠浏览器走查: 渲染后查 .katex-error, 再对 KaTeX 留下的 <annotation> 源码跑同一个 lintTex。

for (const b of bad) console.error(b)
console.log(`${n - bad.length} / ${n} 个公式通过${skipped.length ? `  (跳过 ${skipped.join(', ')})` : ''}`)
process.exit(bad.length ? 1 : 0)
