// node scripts/check-bundle.mjs: 防止一次 eager import 把所有章节重新塞进首页。
import assert from 'node:assert/strict'
import { readFileSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
const dist = fileURLToPath(new URL('../dist/', import.meta.url))
const manifest = JSON.parse(readFileSync(dist + '.vite/manifest.json', 'utf8'))
const entry = manifest['index.html']
const initial = new Set()
function visit(key) {
  if (initial.has(key)) return
  initial.add(key)
  for (const child of manifest[key].imports || []) visit(child)
}
visit('index.html')
for (const stage of ['basic', 'models', 'train', 'finetune', 'infer', 'agent']) {
  for (const kind of ['topics', 'quiz']) {
    const key = `src/data/${kind}/${stage}.js`
    assert(manifest[key]?.isDynamicEntry, `${key} 必须按需加载`)
    assert(!initial.has(key), `${key} 不应进入入口依赖`)
  }
}
const bytes = statSync(dist + entry.file).size
assert(bytes < 200_000, `入口 ${bytes} B 超过 200 kB, 请检查是否误引入正文或题库`)
const total = [...initial].reduce((sum, key) => sum + statSync(dist + manifest[key].file).size, 0)
console.log(`入口 ${(bytes / 1000).toFixed(2)} kB; 静态 JS 依赖合计 ${(total / 1000).toFixed(2)} kB; 六阶段正文和题库均按需加载`)
