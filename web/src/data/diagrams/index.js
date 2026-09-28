// 静态图的数据放在这里, 由 components/dag/DagView.vue 统一渲染。
// 不要再手写 SVG 坐标 —— 坐标由 dagre 算, 改数据不会把版式挤乱。
//
// 只放"画死的"图。跟着页面状态变的图 (比如 Blocks.vue 的 block 接线图要跟着零件槽位走)
// 直接在那个 view 里写 computed, 不要在这儿放一份会过期的副本。
const files = import.meta.glob('./*.js', { eager: true, import: 'default' })
export const diagrams = Object.assign({}, ...Object.values(files).filter(Boolean))
