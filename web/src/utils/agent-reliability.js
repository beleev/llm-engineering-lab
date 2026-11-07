// 教学模型与 Python 对照, 数字由 tests/test_agent_extensions.py 逐项核对。
// 不接真实 API。界面只展示任务状态, 不展示实现配置。
export function memoryAnswers(mode) {
  const facts = [
    { entity: 'alice', scope: '*', value: 'Python' },
    { entity: 'alice', scope: 'A', value: 'TypeScript' },
  ]
  return [
    ['alice', 'A'],
    ['alice', 'B'],
    ['bob', 'A'],
  ].map(([entity, scope]) => {
    const own = facts.filter((f) => f.entity === entity)
    const fact =
      mode === 'append-first'
        ? own[0]
        : mode === 'overwrite'
          ? own.at(-1)
          : own.find((f) => f.scope === scope) || own.find((f) => f.scope === '*')
    return fact?.value || 'unknown'
  })
}

export function completion(mode = 'verified', repairs = 1, stubborn = false, maxCalls = 4) {
  const notes = { existing: 'keep' },
    frames = []
  let calls = 0,
    passed = false
  const required = ['alpha', 'beta', 'gamma']
  for (let attempt = 0; attempt <= (mode === 'verified' ? repairs : 0); attempt++) {
    const remaining = mode === 'verified' ? maxCalls - calls : 2
    if (remaining <= 0) break
    const missing = required.filter((k) => notes[k] !== 'saved')
    calls++
    if (!stubborn) {
      const writes = attempt === 0 ? ['alpha', 'beta'] : missing
      writes.forEach((k) => {
        notes[k] = 'saved'
      })
      frames.push({ text: `写入 ${writes.join('、')}`, count: Object.keys(notes).length - 1 })
      if (remaining > 1) calls++
    }
    frames.push({
      text: stubborn || remaining > 1 ? '模型说：已完成' : '模型调用预算耗尽',
      count: Object.keys(notes).length - 1,
    })
    const actual = required.every((k) => notes[k] === 'saved')
    passed = mode === 'verified' ? actual : true
    frames.push({
      text: mode === 'verified' ? (actual ? '环境验收通过' : '环境验收失败') : '直接接受完成声明',
      count: Object.keys(notes).length - 1,
    })
    if (passed) break
  }
  return { passed, actual: required.every((k) => notes[k] === 'saved'), calls, notes, frames }
}

export function retryWrite(mode, allowed = true, attempts = 2) {
  let writes = 0,
    requests = 0,
    queries = 0,
    receipt = false,
    lost = false,
    status = 'unknown'
  const frames = []
  for (let i = 0; i < attempts; i++) {
    requests++
    if (!allowed) {
      status = 'denied'
      frames.push('授权拒绝，停止重试')
      break
    }
    if (mode !== 'blind' && receipt) {
      status = 'confirmed'
      frames.push('同一个幂等键，返回原收据')
      break
    }
    writes++
    receipt = mode !== 'blind'
    frames.push('服务端已写入 10 元')
    if (!lost) {
      lost = true
      frames.push('响应丢失，客户端不知道是否成功')
      if (mode === 'query') {
        queries++
        status = 'confirmed'
        frames.push('查询收据，确认已经成功')
        break
      }
    } else {
      status = 'confirmed'
      break
    }
  }
  return { writes, amount: writes * 10, requests, queries, status, frames }
}

export function schedule(mode, budget = 30) {
  const jobs = [
    { id: 'fast-wrong', ready: 1, valid: false },
    { id: 'correct', ready: 3, valid: true },
    { id: 'slow', ready: 6, valid: true },
  ].map((j) => ({ ...j, state: 'running' }))
  const accepted = [],
    frames = []
  let work = 0,
    elapsed = 0
  for (let tick = 1; tick <= 6; tick++) {
    for (const job of jobs) {
      if (job.state !== 'running') continue
      if (work >= budget) {
        job.state = 'canceled'
        continue
      }
      work++
      elapsed = tick
      if (tick >= job.ready) {
        job.state = 'completed'
        if (job.valid) accepted.push(job.id)
      }
    }
    if (mode === 'first-valid' && accepted.length)
      jobs.forEach((j) => {
        if (j.state === 'running') j.state = 'canceled'
      })
    frames.push({ tick, work, states: Object.fromEntries(jobs.map((j) => [j.id, j.state])) })
    if (mode === 'first-valid' && accepted.length) break
  }
  return { accepted, work, elapsed, states: Object.fromEntries(jobs.map((j) => [j.id, j.state])), frames }
}

export function contextCache(turns = 6, placement = 'tail', threshold = 0, keepScope = true) {
  const entries = new Map(),
    usage = { read: 0, write: 0, input: 0 },
    rows = []
  let history = [],
    cost = 0,
    summaryCost = 0,
    summaries = 0,
    peak = 0,
    retained = 3
  const policy = ['policy', 200]
  function charge(blocks, now) {
    const prefixes = blocks.map((_, i) => JSON.stringify(blocks.slice(0, i + 1)))
    const breakpoints = [0, blocks.length - 1]
    let hit = 0
    for (const bp of breakpoints)
      for (let i = bp; i > Math.max(-1, bp - 20); i--) {
        if ((entries.get(prefixes[i]) ?? -1) > now) {
          hit = Math.max(hit, i + 1)
          break
        }
      }
    if (hit) entries.set(prefixes[hit - 1], now + 300)
    breakpoints.forEach((bp) => {
      if (bp + 1 > hit) entries.set(prefixes[bp], now + 300)
    })
    const read = blocks.slice(0, hit).reduce((s, b) => s + b[1], 0)
    const write = blocks.slice(hit).reduce((s, b) => s + b[1], 0)
    usage.read += read
    usage.write += write
    return read * 0.1 + write * 1.25
  }
  for (let turn = 0; turn < turns; turn++) {
    if (threshold && history.length >= threshold) {
      const expense = charge([policy, ...history, ['summarize', 10]], turn * 10) + 50
      cost += expense
      summaryCost += expense
      summaries++
      if (!keepScope) retained = 1
      history = [[`summary:${retained}`, 50]]
    }
    const state = [`memory:${retained};done:${turn}`, 40],
      query = [`user:${turn}`, 20]
    const blocks = placement === 'front' ? [state, policy, ...history, query] : [policy, ...history, state, query]
    const amount = charge(blocks, turn * 10 + 1),
      input = blocks.reduce((s, b) => s + b[1], 0)
    cost += amount
    peak = Math.max(peak, input)
    rows.push({ turn: turn + 1, input, cost: +amount.toFixed(2) })
    history.push(query, [`tool:${turn}`, 200])
  }
  return {
    cost: +cost.toFixed(2),
    summary_cost: +summaryCost.toFixed(2),
    summaries,
    peak,
    retained,
    audit_blocks: turns * 2,
    usage,
    rows,
  }
}

export const ragDocs = [
  {
    id: 'alias-comet',
    entity: 'atlas',
    version: 'v2',
    body: 'comet project maps to atlas v2',
    alias: 'comet',
    limit: null,
  },
  { id: 'atlas-v1', entity: 'atlas', version: 'v1', body: 'upload limit is 10 MB', alias: '', limit: 10 },
  { id: 'atlas-v2', entity: 'atlas', version: 'v2', body: 'upload limit is 20 MB', alias: '', limit: 20 },
  { id: 'orion-v2', entity: 'orion', version: 'v2', body: 'upload limit is 90 MB', alias: '', limit: 90 },
]
export const ragQuestions = [
  { id: 'direct', query: 'atlas v2 upload limit', entity: 'atlas', version: 'v2', alias: '' },
  { id: 'two-hop', query: 'comet upload limit', entity: '', version: '', alias: 'comet' },
  { id: 'missing', query: 'nova upload limit', entity: '', version: '', alias: 'nova' },
]
function retrieve(query, contextual) {
  const words = (s) => s.toLowerCase().match(/[a-z0-9]+/g) || []
  const counts = (s) => words(s).reduce((m, w) => ((m[w] = (m[w] || 0) + 1), m), {})
  const docs = ragDocs.map((d, i) => counts(`chunk${i} ${contextual ? `${d.entity} ${d.version} ` : ''}${d.body}`))
  const df = {}
  docs.forEach((d) =>
    Object.keys(d).forEach((w) => {
      df[w] = (df[w] || 0) + 1
    }),
  )
  function vector(tf) {
    const v = Object.fromEntries(
      Object.entries(tf)
        .filter(([w]) => df[w])
        .map(([w, n]) => [w, (1 + Math.log(n)) * Math.log(1 + (4 - df[w] + 0.5) / (df[w] + 0.5))]),
    )
    const norm = Math.hypot(...Object.values(v)) || 1
    return Object.fromEntries(Object.entries(v).map(([w, n]) => [w, n / norm]))
  }
  const q = vector(counts(query))
  return docs
    .map((d, i) => ({ i, score: Object.entries(vector(d)).reduce((s, [w, n]) => s + n * (q[w] || 0), 0) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score || b.i - a.i)
    .map((r) => ragDocs[r.i])
}
export function evidenceChain(index, mode = 'agentic', budget = 2) {
  const q = ragQuestions[index],
    sources = [],
    queries = []
  let entity = q.entity,
    version = q.version,
    value = null
  if (budget) {
    queries.push(q.query)
    let hits = retrieve(q.query, mode !== 'plain')
    if (q.alias) {
      const alias = hits.find((d) => d.alias === q.alias)
      if (alias) {
        entity = alias.entity
        version = alias.version
        sources.push(alias.id)
      }
      if (mode === 'agentic' && alias && queries.length < budget) {
        const query = `${entity} ${version} upload limit`
        queries.push(query)
        hits = retrieve(query, true)
      } else hits = []
    }
    const doc = (mode === 'plain' ? hits.slice(0, 1) : hits).find(
      (d) => d.limit !== null && d.entity === entity && d.version === version,
    )
    if (doc) {
      value = doc.limit
      sources.push(doc.id)
    }
  }
  return { answer: value, sources, queries, status: value === null ? 'insufficient' : 'answered' }
}

export const trajectories = [
  { name: 'verified', actions: ['lookup', 'answer:observed'] },
  { name: 'lucky-guess', actions: ['answer:10'] },
  { name: 'forbidden', actions: ['steal', 'answer:observed'] },
  { name: 'over-budget', actions: ['lookup', 'lookup', 'lookup', 'answer:observed'] },
  { name: 'wrong', actions: ['lookup', 'answer:11'] },
]
export function trajectoryReport(actions, budget = 3) {
  let observed = null,
    final = null,
    authorized = true
  const reasons = actions.map((action, i) => {
    if (final !== null) return 'after-final'
    if (i >= budget) return 'budget'
    if (action === 'lookup') {
      observed = 10
      return ''
    }
    if (action === 'steal') {
      observed = 10
      authorized = false
      return 'authorization'
    }
    if (action.startsWith('answer:')) {
      const raw = action.split(':')[1]
      final = raw === 'observed' ? observed : Number(raw)
      if (observed === null || !authorized) return 'unsupported'
      return final === 10 ? '' : 'wrong-result'
    }
    return 'unknown-action'
  })
  const result = final === 10,
    process = reasons.length > 0 && reasons.every((r) => !r) && final !== null
  return {
    result,
    process,
    eligible: result && process,
    step_labels: reasons.map((r) => !r),
    reasons,
    calls: actions.length,
  }
}

export const learningTasks = [
  ['L1', 'learning', 'alice', 'A', 'TypeScript', ['A', 'TypeScript']],
  ['L2', 'learning', 'alice', 'B', 'Python', ['*', 'Python']],
  ['T1', 'transfer', 'alice', 'A', 'TypeScript'],
  ['T2', 'transfer', 'alice', 'B', 'Python'],
  ['T3', 'transfer', 'bob', 'A', 'unknown'],
  ['C1', 'change', 'alice', 'A', 'Rust', ['A', 'Rust']],
  ['C2', 'change', 'alice', 'A', 'Rust'],
  ['R1', 'retention', 'alice', 'B', 'Python'],
  ['R2', 'retention', 'alice', 'A', 'Rust'],
]
export function learning(mode) {
  const memory = new Map(),
    notes = [],
    rows = [],
    order = []
  learningTasks.forEach(([id, phase, entity, scope, expected, feedback]) => {
    const recalled =
      mode === 'versioned'
        ? memory.get(`${entity}:${scope}`) || memory.get(`${entity}:*`)
        : mode === 'append'
          ? notes.find((n) => n.entity === entity)?.value
          : undefined
    const action = recalled || (entity === 'alice' ? 'Python' : 'unknown')
    order.push(`${id}:act`, `${id}:grade`)
    const updated = !!feedback && mode !== 'static'
    if (updated) {
      notes.push({ entity, value: feedback[1] })
      memory.set(`${entity}:${feedback[0]}`, feedback[1])
      order.push(`${id}:update`)
    }
    rows.push({ id, phase, action, expected, passed: action === expected, activated: !!recalled, updated })
  })
  const scores = Object.fromEntries(
    ['learning', 'transfer', 'change', 'retention'].map((phase) => {
      const subset = rows.filter((r) => r.phase === phase)
      return [phase, subset.filter((r) => r.passed).length / subset.length]
    }),
  )
  return {
    rows,
    order,
    scores,
    negative_transfer: rows.filter((r) => r.phase !== 'learning' && r.activated && !r.passed).length,
    updates: rows.filter((r) => r.updated).length,
    activated: rows.filter((r) => r.activated).length,
  }
}
