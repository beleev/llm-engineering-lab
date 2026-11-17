// richText.js 的切分规则自检: 括号注解、代码标识符、关键数字、首句。npm run check:richtext
import assert from 'node:assert/strict'
import { richTokens, splitLead } from '../src/utils/richText.js'
import { zhPunct } from '../src/utils/zhPunct.js'

const kinds = (s) => richTokens(s).filter((t) => t.kind).map((t) => `${t.kind}:${t.text}`)

// 括号里有中文才算注解; 函数调用和纯英文括号不算
assert.deepEqual(kinds('GQA (几个 query 头共用一组 K 和 V) 省显存'), ['gloss:(几个 query 头共用一组 K 和 V)'])
assert.deepEqual(kinds('调用 next(messages) 返回'), [])
assert.deepEqual(kinds('精度 (fp16) 下'), [])
// 注解内部的代码和数字照样标出来
assert.deepEqual(richTokens('块 (每块 bs 个 token, 共 384 个)')[1].parts.filter((t) => t.kind).map((t) => t.text), ['384'])
// 代码: 带 _ / . / 以 () 结尾; 普通缩写、型号、版本号不算
assert.deepEqual(kinds('tool_use 和 Agent.run 和 next() 和 llm_agent/m01_agent_loop/demo.py'), ['code:tool_use', 'code:Agent.run', 'code:next()', 'code:llm_agent/m01_agent_loop/demo.py'])
assert.deepEqual(kinds('KV cache、K/V、LoRA、Qwen2.5、LLaMA-7B、m01'), [])
// 数字: 两位以上 / 小数 / 科学计数 / % / ×; 单个数字和标识符里的数字不算
assert.deepEqual(kinds('第 3 步从 688 降到 37 (18.6×), 误差 2.80e-06, 20~40%'), ['num:688', 'num:37', 'num:18.6×', 'num:2.80e-06', 'num:20', 'num:40%'])
assert.deepEqual(kinds('fp16、m02、GPT-3'), [])
// 首句: 公式里的句号不算
assert.deepEqual(splitLead('先说结论。再说机制。'), ['先说结论。', '再说机制。'])
assert.deepEqual(splitLead('式子 $a。b$ 成立。后面'), ['式子 $a。b$ 成立。', '后面'])
assert.deepEqual(splitLead('没有句号'), ['没有句号', ''])
// 拼回去必须和原文一字不差
for (const s of ['a (中文 b_c 12) d.e 3.5× 。', ' (开头)', '尾巴 (x 中)']) {
  assert.equal(richTokens(s).map((t) => t.text).join(''), s)
}
// 中文标点: 正文转换, 代码不动
assert.equal(zhPunct('分页 (PagedAttention) 借的是, 结论: 省显存; 对吗?'), '分页（PagedAttention）借的是，结论：省显存；对吗？')
assert.equal(zhPunct('调用 next(messages, tool_schemas) 后'), '调用 next(messages, tool_schemas) 后')
assert.equal(zhPunct('地址是 pool[table[pos // bs], pos % bs], 对'), '地址是 pool[table[pos // bs], pos % bs]，对')
assert.equal(zhPunct('块 {type: tool_use, id} 和 1:9、12:30、https://a.b 和 1,000'), '块 {type: tool_use, id} 和 1:9、12:30、https://a.b 和 1,000')
assert.equal(zhPunct('(原点) 每个数'), '（原点）每个数')
assert.equal(zhPunct('pure english, no change: ok?'), 'pure english, no change: ok?')
assert.equal(zhPunct('关键词: '), '关键词：')
assert.equal(zhPunct(42), 42)
assert.equal(zhPunct(zhPunct('a, 中文 (b, c)')), zhPunct('a, 中文 (b, c)')) // 幂等
// 转换后的中文括号照样识别成注解
assert.deepEqual(kinds(zhPunct('GQA (几个 query 头共用一组 K 和 V) 省显存')), ['gloss:（几个 query 头共用一组 K 和 V）'])
console.log('richText / zhPunct: 全部通过')
