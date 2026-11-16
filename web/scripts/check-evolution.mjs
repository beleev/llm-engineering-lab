// 每章都要有「演进逻辑链」, 而且要短: 3–5 步, 每步痛点和解法各一句。  用法: npm run check:evolution
// 章节正文在 data/topics/<阶段>.js 的 pages[route].evolution; 手写视图的演进链不在这里查。
// 研究 / 工程实验回放 (docs/*-completion-manifest.json 每条 entry 的 evolution) 用同一套规则查。
import { readdirSync, readFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const dir = join(here, '../src/data/topics')
const len = (s) => s.replace(/\$([^$]*)\$/g, (_, t) => 'x'.repeat(Math.ceil(t.length / 2))).length  // 公式按源码长度的一半计
const MAX_PAIN = 50, MAX_FIX = 60
const bad = []
const check = (e, where) => {
  if (!e) { bad.push(`✗ ${where}: 没有 evolution`); return }
  if (!e.title || !e.subtitle) bad.push(`✗ ${where}: 缺 title 或 subtitle`)
  if (!(e.steps?.length >= 3 && e.steps.length <= 5)) bad.push(`✗ ${where}: 步数 ${e.steps?.length}, 要 3–5 步`)
  ;(e.steps || []).forEach((s, i) => {
    if (!s.name || !s.pain || !s.fix) bad.push(`✗ ${where}[${i}]: name / pain / fix 缺一`)
    if (s.pain && len(s.pain) > MAX_PAIN) bad.push(`✗ ${where}[${i}]: 痛点 ${len(s.pain)} 字 > ${MAX_PAIN}`)
    if (s.fix && len(s.fix) > MAX_FIX) bad.push(`✗ ${where}[${i}]: 解法 ${len(s.fix)} 字 > ${MAX_FIX}`)
  })
}

let total = 0
for (const f of readdirSync(dir).filter((f) => f.endsWith('.js') && f !== 'index.js')) {
  const pages = (await import(pathToFileURL(join(dir, f)))).default.pages || {}
  for (const [route, page] of Object.entries(pages)) {
    if (['attention', 'position'].includes(route)) continue // 主讲链在手写视图中
    total++
    check(page.evolution, `${f} › ${route}`)
  }
}
const counts = []
for (const [label, file] of [['研究实验', 'research'], ['工程实验', 'engineering']]) {
  const name = `${file}-completion-manifest.json`
  const { entries } = JSON.parse(readFileSync(join(here, '../../docs', name), 'utf8'))
  for (const e of entries) check(e.evolution, `${name} › ${e.id}`)
  counts.push(`${label} ${entries.length} 组`)
}
for (const b of bad) console.error(b)
console.log(`${total} 章; ${counts.join('; ')}; 不合格 ${bad.length} 处`)
process.exit(bad.length ? 1 : 0)
