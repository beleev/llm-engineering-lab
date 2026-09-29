// 只把目录和阅读量送到首页; 正文、题库和 .vue 源码留在各自的分包或构建进程里。
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { join } from 'node:path'

const src = fileURLToPath(new URL('../src/', import.meta.url))
const id = 'virtual:course-catalog'
const resolved = '\0' + id
const textLen = (s) => (s.match(/[一-龥]/g) || []).length
  + (s.match(/[A-Za-z][A-Za-z0-9_.]*/g) || []).length

export default function courseCatalog() {
  return {
    name: 'course-catalog',
    resolveId: (name) => name === id ? resolved : null,
    async load(name) {
      if (name !== resolved) return
      const extraChapters = {}, pageStats = {}, viewStats = {}, quizStages = {}
      for (const kind of ['topics', 'quiz']) {
        const dir = join(src, 'data', kind)
        for (const file of readdirSync(dir).filter((f) => f.endsWith('.js') && f !== 'index.js')) {
          const path = join(dir, file)
          this.addWatchFile(path)
          const { default: mod } = await import(pathToFileURL(path).href + '?catalog=' + Date.now())
          if (kind === 'quiz') {
            for (const route of Object.keys(mod)) quizStages[route] = file.slice(0, -3)
            continue
          }
          extraChapters[mod.stage] = mod.chapters || []
          for (const [route, page] of Object.entries(mod.pages || {})) {
            const parts = [page.subtitle, page.tldr, page.question,
              ...(page.points || []).flatMap((p) => [p.title, p.body]),
              ...(page.links || []).map((l) => l.body),
              ...(page.sourceRows || []).map((r) => r.takeaway), page.snippet]
            pageStats[route] = { chars: textLen(parts.filter(Boolean).join(' ')), widgets: page.widgets || [] }
          }
        }
      }
      for (const file of readdirSync(join(src, 'views')).filter((f) => f.endsWith('.vue'))) {
        const path = join(src, 'views', file)
        this.addWatchFile(path)
        const text = readFileSync(path, 'utf8')
        const body = text.replace(/<script[\s\S]*?<\/script>/g, '')
          .replace(/<style[\s\S]*?<\/style>/g, '').replace(/<[^>]+>/g, ' ')
        viewStats[file.slice(0, -4)] = {
          chars: textLen(body),
          labs: new Set([...text.matchAll(/components\/labs\/(\w+)\.vue/g)].map((m) => m[1])).size,
        }
      }
      return Object.entries({ extraChapters, pageStats, viewStats, quizStages })
        .map(([key, value]) => `export const ${key} = ${JSON.stringify(value)}`).join('\n')
    },
    handleHotUpdate({ file, server }) {
      if (!file.startsWith(src) || !/\/(data\/(topics|quiz)|views)\//.test(file)) return
      const mod = server.moduleGraph.getModuleById(resolved)
      if (mod) server.moduleGraph.invalidateModule(mod)
      server.ws.send({ type: 'full-reload' })
      return []
    },
  }
}
