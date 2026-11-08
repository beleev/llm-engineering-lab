// 按主题组合进阶段落；目录、正文、题库使用同一归属映射。
import { chapterAliases } from '../data/chapterAliases.js'

export function mergePage(topic, from, to, title, labNames) {
  const page = topic.pages[from]
  if (!page || !topic.pages[to]) throw new Error(`无法合并章节: ${from} → ${to}`)
  const { anchor } = chapterAliases[from]
  ;(topic.pages[to].sections ||= []).push({ ...page, id: anchor, title, labNames })
  const links = [...(topic.pages[to].related || []), ...(page.related || [])]
  topic.pages[to].related = links.filter((link, i) => (chapterAliases[link.route]?.route || link.route) !== to && links.findIndex(other => other.route === link.route) === i)
  delete topic.pages[from]
  topic.chapters = topic.chapters.filter(c => c.route !== from)
}

export function mergeQuiz(bank) {
  for (const [from, { route }] of Object.entries(chapterAliases)) {
    if (!bank[from]) continue
    bank[route] = [...(bank[route] || []), ...bank[from]]
    delete bank[from]
  }
  return bank
}

export function canonicalizeRelated(topic) {
  for (const page of Object.values(topic.pages)) {
    for (const part of [page, ...(page.sections || [])]) {
      for (const link of part.related || []) link.route = chapterAliases[link.route]?.route || link.route
    }
  }
}
