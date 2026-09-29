// 路由守卫加载当前章节所在的题库, 不在首页下载全部题目。
import { shallowReactive } from 'vue'
import { quizStages } from 'virtual:course-catalog'

const loaders = import.meta.glob(['./*.js', '!./index.js'], { import: 'default' })
const loaded = new Map()
export const quizBank = shallowReactive({})

export async function loadQuiz(route) {
  const stage = quizStages[route]
  if (!stage) return
  if (!loaded.has(stage)) {
    loaded.set(stage, loaders[`./${stage}.js`]().then((items) => Object.assign(quizBank, items)).catch((error) => {
      loaded.delete(stage)
      throw error
    }))
  }
  await loaded.get(stage)
}
