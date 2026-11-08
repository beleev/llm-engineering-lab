import assert from 'node:assert/strict'
import { chapterAliases, migrateProgress } from '../src/data/chapterAliases.js'
import { chapterGroups } from '../src/data/chapterGroups.js'
const stages = ['basic','models','train','finetune','infer','agent']
const pages = {}, quizzes = {}
for (const stage of stages) {
  Object.assign(pages, (await import(`../src/data/topics/${stage}.js`)).default.pages)
  Object.assign(quizzes, (await import(`../src/data/quiz/${stage}.js`)).default)
}
const routes = Object.values(chapterGroups).flatMap(groups => groups.flatMap(g => g.routes))
assert.equal(routes.length, 101)
assert.equal(new Set(routes).size, routes.length)
for (const [old, alias] of Object.entries(chapterAliases)) {
  assert(!routes.includes(old))
  assert(!pages[old] && !quizzes[old])
  assert(pages[alias.route].sections.some(s => s.id === alias.anchor))
}
assert.equal(Object.values(quizzes).reduce((n,q) => n+q.length,0),396)
const original={ visited:{'finetune-dora':2,'finetune-lora':1}, last:'finetune-dora', quiz:{'finetune-dora':{correct:3,total:3}} }
const migrated=migrateProgress(original)
assert.equal(migrated.last,'finetune-lora')
assert.equal(migrated.visited['finetune-lora'],2)
assert(!migrated.quiz['finetune-lora'])
assert.equal(migrated.archivedQuiz['finetune-dora'].correct,3)
assert.deepEqual(migrateProgress(migrated),migrated)
assert.equal(original.last,'finetune-dora')
for (const [route,page] of Object.entries(pages)) {
  const ids=(page.sections || []).map(s=>s.id)
  assert.equal(ids.length,new Set(ids).size,route)
  for(const part of [page,...(page.sections || [])]) for(const ref of part.related || []) assert(routes.includes(ref.route),ref.route)
}
assert(pages['models-mtp'].points.every(p => !/状态|mask|深度/.test(p.title)))
assert(!pages['models-dsa'].links.some(l=>l.from==='models-mtp · 掩码实验台'))
console.log('PASS: 101 章、396 题、6 个旧入口、进度归档与段落归属')
