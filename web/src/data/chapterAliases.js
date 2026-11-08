// 旧书签保持可达；目录只列主讲章。锚点定位到原章节迁入的段落。
export const chapterAliases = {
  'train-low-precision': { route: 'train-precision-stability', anchor: 'block-scale' },
  'finetune-dora': { route: 'finetune-lora', anchor: 'dora' },
  'agent-context-cost': { route: 'agent-prompt-caching', anchor: 'context-cost' },
  'agent-collaboration': { route: 'agent-orchestrator', anchor: 'collaboration' },
  'agent-memory-privacy': { route: 'agent-memory-lifecycle', anchor: 'memory-privacy' },
  'models-qknorm-yarn': { route: 'attention', anchor: 'qk-norm' },
}
export const canonicalChapter = route => chapterAliases[route]?.route || route

// 合并后题数和教学范围改变，旧成绩不能直接换算成新章“全部掌握”。
// 保留原成绩供追溯，清掉受影响目标成绩，让用户重新自测。
export function migrateProgress(saved) {
  const state = { ...saved, visited: { ...(saved.visited || {}) }, quiz: { ...(saved.quiz || {}) },
    archivedQuiz: { ...(saved.archivedQuiz || {}) } }
  if (state.curriculumVersion === 2) return state
  const targets = new Set(['position', 'attention', 'models-mtp', 'models-mamba'])
  for (const [old, alias] of Object.entries(chapterAliases)) {
    if (state.visited[old]) state.visited[alias.route] = Math.max(state.visited[alias.route] || 0, state.visited[old])
    delete state.visited[old]
    if (state.quiz[old]) state.archivedQuiz[old] = state.quiz[old]
    delete state.quiz[old]
    targets.add(alias.route)
  }
  for (const target of targets) {
    if (state.quiz[target]) state.archivedQuiz[target] = state.quiz[target]
    delete state.quiz[target]
  }
  state.last = canonicalChapter(state.last)
  state.curriculumVersion = 2
  return state
}
