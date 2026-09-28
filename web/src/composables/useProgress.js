// 学习进度: 打开过哪些章、自测最好答对几道、选的阅读档位。只存在本机 localStorage, 不上传。
import { reactive, watch } from 'vue'

const KEY = 'llm-progress'
const load = () => {
  try { return JSON.parse(localStorage.getItem(KEY)) || {} } catch (_) { return {} }
}
const saved = load()
const state = reactive({ visited: saved.visited || {}, quiz: saved.quiz || {}, last: saved.last || null, level: saved.level || 'all' })

watch(state, (s) => {
  try { localStorage.setItem(KEY, JSON.stringify(s)) } catch (_) { /* 隐私模式下静默降级 */ }
}, { deep: true })

export function useProgress() {
  return {
    state,
    visit(route) {
      if (!route) return
      state.visited[route] = Date.now()
      // 首页 / 术语表 / 对照表 是"中转站", 不算学习断点, 否则回首页就把断点冲掉了
      if (!['home', 'glossary', 'compare'].includes(route)) state.last = route
    },
    isVisited: (route) => !!state.visited[route],
    // 记的是最好成绩: 重做时答得更差不覆盖, 侧栏的 ✓ 不会因为重做错一题就消失。
    // 题目数量变了 (total 不同) 说明题库换过, 成绩重新记。
    setQuiz(route, correct, total) {
      const old = state.quiz[route]
      if (!old || old.total !== total || correct > old.correct) state.quiz[route] = { correct, total }
    },
    quizOf: (route) => state.quiz[route] || null,
    isMastered: (route) => { const q = state.quiz[route]; return !!q && q.correct === q.total },
    setLevel(l) { state.level = l },
    reset() { state.visited = {}; state.quiz = {}; state.last = null },
  }
}
