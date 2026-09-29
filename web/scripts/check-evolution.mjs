// 每章都要有「演进逻辑链」, 而且要短: 3–5 步, 每步痛点和解法各一句。  用法: npm run check:evolution
// 章节正文在 data/topics/<阶段>.js 的 pages[route].evolution; 手写视图的演进链不在这里查。
import { readdirSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'

const dir = join(dirname(fileURLToPath(import.meta.url)), '../src/data/topics')
const len = (s) => s.replace(/\$([^$]*)\$/g, (_, t) => 'x'.repeat(Math.ceil(t.length / 2))).length  // 公式按源码长度的一半计
const MAX_PAIN = 50, MAX_FIX = 60
let total = 0
const bad = []
for (const f of readdirSync(dir).filter((f) => f.endsWith('.js') && f !== 'index.js')) {
  const pages = (await import(pathToFileURL(join(dir, f)))).default.pages || {}
  for (const [route, page] of Object.entries(pages)) {
    total++
    const e = page.evolution, where = `${f} › ${route}`
    if (!e) { bad.push(`✗ ${where}: 没有 evolution`); continue }
    if (!e.title || !e.subtitle) bad.push(`✗ ${where}: 缺 title 或 subtitle`)
    if (!(e.steps?.length >= 3 && e.steps.length <= 5)) bad.push(`✗ ${where}: 步数 ${e.steps?.length}, 要 3–5 步`)
    ;(e.steps || []).forEach((s, i) => {
      if (!s.name || !s.pain || !s.fix) bad.push(`✗ ${where}[${i}]: name / pain / fix 缺一`)
      if (s.pain && len(s.pain) > MAX_PAIN) bad.push(`✗ ${where}[${i}]: 痛点 ${len(s.pain)} 字 > ${MAX_PAIN}`)
      if (s.fix && len(s.fix) > MAX_FIX) bad.push(`✗ ${where}[${i}]: 解法 ${len(s.fix)} 字 > ${MAX_FIX}`)
    })
  }
}
for (const b of bad) console.error(b)
console.log(`${total} 章; 不合格 ${bad.length} 处`)
process.exit(bad.length ? 1 : 0)
