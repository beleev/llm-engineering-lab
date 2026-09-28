// 自测题不能靠长度蒙对: 正确项比次长的选项长 30% 以上就报错 (公式按源码长度的一半计)。
// 出题时正确项容易写得更具体、更长, 读者不懂也能挑最长的那个。  用法: npm run check:quiz
import { readdirSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'

const dir = join(dirname(fileURLToPath(import.meta.url)), '../src/data/quiz')
const len = (s) => s.replace(/\$([^$]*)\$/g, (_, t) => 'x'.repeat(Math.ceil(t.length / 2))).length
let total = 0, longest = 0
const bad = []
for (const f of readdirSync(dir).filter((f) => f.endsWith('.js') && f !== 'index.js')) {
  const quiz = (await import(pathToFileURL(join(dir, f)))).default
  for (const [route, qs] of Object.entries(quiz)) qs.forEach((q, i) => {
    total++
    const L = q.options.map(len), right = L[q.answer], other = Math.max(...L.filter((_, j) => j !== q.answer))
    if (right === Math.max(...L)) longest++
    if (right >= 1.3 * other) bad.push(`✗ ${f} › ${route}[${i}]: 正确项 ${right} 字, 次长 ${other} 字`)
  })
}
for (const b of bad) console.error(b)
console.log(`${total} 道题; 正确项恰好最长 ${longest} 道 (${Math.round((100 * longest) / total)}%); 长度露馅 ${bad.length} 道`)
process.exit(bad.length ? 1 : 0)
