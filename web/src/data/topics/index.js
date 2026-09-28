// 按阶段拆分的章节扩展: 本目录下每个阶段一个文件, 形如
//   export default {
//     stage: 'train',                                   // basic | models | train | finetune | infer | agent
//     chapters: [{ route: 'train-muon', label: 'Muon 优化器', hint: '…' }],   // 追加到该阶段侧栏末尾;
//                                                   加 after: '某章 route' 就插到那一章后面
//     pages: { 'train-muon': { title, subtitle, tldr, question, code, points, links, sourceRows,
//                              snippetTitle, snippet, source, run, widgets } },
//   }
// 路由 (/阶段/章节名)、侧栏、learningPath 全部由这里自动生成, 新增一章不需要再改 router 和 models.js。
// 章节正文只住在 pages 里: models.js 的 topicPages 就是把各阶段的 pages 合并起来, 它自己不存正文。
// 同一个 route 名在两个文件里都写了 page 时, 后加载的整页覆盖先加载的, 所以一个 route 只写一处。
const files = import.meta.glob('./*.js', { eager: true, import: 'default' })
const mods = Object.entries(files).filter(([f]) => !f.endsWith('/index.js')).map(([, m]) => m).filter(Boolean)

export const extraChapters = {}
export const extraPages = {}
for (const m of mods) {
  extraChapters[m.stage] = [...(extraChapters[m.stage] || []), ...(m.chapters || [])]
  Object.assign(extraPages, m.pages || {})
}
