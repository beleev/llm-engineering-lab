// 目录由构建插件提取; 进入一个阶段时才下载正文。新增章节仍只改该阶段文件。
import { shallowReactive } from 'vue'
export { extraChapters } from 'virtual:course-catalog'

const loaders = import.meta.glob(['./*.js', '!./index.js'], { import: 'default' })
const loaded = new Map()
export const topicPages = shallowReactive({})

export async function loadTopics(stage) {
  const loader = loaders[`./${stage}.js`]
  if (!loader) return
  if (!loaded.has(stage)) {
    loaded.set(stage, loader().then((mod) => Object.assign(topicPages, mod.pages)).catch((error) => {
      loaded.delete(stage) // 网络恢复后可以重试
      throw error
    }))
  }
  await loaded.get(stage)
}
