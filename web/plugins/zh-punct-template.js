// Vue 模板编译期的中文标点: 模板里的静态文字直接转换 (零运行时开销), {{ 插值 }} 包一层 $zh(...) 在运行时转换。
// 转换规则见 src/utils/zhPunct.js; 数据和源码里照旧写半角, 只有页面显示是中文标点。
// 不转换: <code> <pre> <kbd> <samp> <textarea> <script> <style> 里面、类名是 code / cm / codeblock / endpoint / block 的元素里面、
// 源码对照表的代码列 (td.mono)、v-pre 里面,
// 以及 Tex.vue (它自己先切公式再转换, 不能整段转)。
import { zhPunct } from '../src/utils/zhPunct.js'

const SKIP_TAGS = new Set(['code', 'pre', 'kbd', 'samp', 'textarea', 'script', 'style'])
const ELEMENT = 1, TEXT = 2, INTERPOLATION = 5, ATTRIBUTE = 6, DIRECTIVE = 7

// 类名精确匹配: 代码块 / 注释 / 链接端点 / 模拟的消息 (agent transcript) 里是数据, 不是正文
const SKIP_CLASS = new Set(['code', 'cm', 'codeblock', 'endpoint', 'block'])
const classes = (el) => (el.props?.find((p) => p.type === ATTRIBUTE && p.name === 'class')?.value?.content || '').split(/\s+/)
const skipEl = (el) => SKIP_TAGS.has(el.tag)
  || classes(el).some((c) => SKIP_CLASS.has(c))
  || (el.tag === 'td' && classes(el).includes('mono')) // 源码对照表的「代码位置」一列
  || el.props?.some((p) => p.type === DIRECTIVE && p.name === 'pre')

export function zhPunctTransform(node, context) {
  if (/[\\/](Tex)\.vue$/.test(context.filename || '')) return
  if (node.type !== ELEMENT && node.type !== 0 /* ROOT */) return
  // 编译器的节点没有父指针: 要跳过的元素把标记传给子节点, 子节点进来时再往下传, 整棵子树都不动
  if (node.zhSkip || (node.type === ELEMENT && skipEl(node))) {
    for (const child of node.children || []) child.zhSkip = true
    return
  }
  for (const child of node.children || []) {
    if (child.type === TEXT) child.content = zhPunct(child.content)
    else if (child.type === INTERPOLATION && child.content?.type === 4 /* SIMPLE_EXPRESSION */ && !child.content.isStatic) {
      child.content.content = `$zh(${child.content.content})`
      child.content.ast = undefined // 解析阶段按原文预解析过的语法树作废, 让编译器按新表达式重新解析
    }
  }
}
