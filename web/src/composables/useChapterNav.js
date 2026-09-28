// 上一章 / 下一章: 全站只在这里算一次, 侧栏档位、键盘 ← →、章首章尾的链接用的是同一份。
//   const { prev, next, inPath, levelLabel } = useChapterNav()
// 顺序来自 learningPath, 再按当前阅读档位过滤: 选了「冲刺」就只在 12 个冲刺章之间翻。
// 当前所在章始终留在列表里, 所以从档位之外的章出发也能翻到最近的档位内章节。
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { learningPath } from '@/data/models.js'
import { LEVELS, inLevel } from '@/data/tiers.js'
import { useProgress } from '@/composables/useProgress.js'

export function useChapterNav() {
  const route = useRoute()
  const { state } = useProgress()
  const path = computed(() => learningPath.filter((p) => inLevel(p.route, state.level) || p.route === route.name))
  const at = computed(() => path.value.findIndex((p) => p.route === route.name))
  const toNav = (p) => (p ? { name: p.route, label: p.label } : null)
  return {
    path,
    inPath: computed(() => at.value >= 0),                       // 当前路由在不在学习路径里
    prev: computed(() => (at.value > 0 ? toNav(path.value[at.value - 1]) : null)),
    next: computed(() => (at.value >= 0 ? toNav(path.value[at.value + 1]) : null)),
    // 档位不是「全部」时给个名字, 让读者知道翻章为什么会跳着走
    levelLabel: computed(() => (state.level === 'all' ? '' : LEVELS.find((l) => l.id === state.level)?.label || '')),
  }
}
