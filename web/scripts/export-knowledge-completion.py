"""从实际 Python 实验导出网页回放；--check 重跑全部情形并检查源码摘要与结果。"""
from pathlib import Path
import hashlib
import importlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'web/src/data/knowledge-completion.json'
CASES=[
 ('chat','多轮消息: 谁的 token 在算 loss','llm_finetune.run_finetune.chat_template.demo',
  [('correct','正确模板'),('wrong-role','工具角色错配'),('truncated','截断半条消息')],
  ['finetune-sft'], 'loss mask 为什么不能拿来当 attention mask?','两个 mask 管的事不一样: loss mask 决定哪些 token 当训练目标, attention mask 决定模型能看到哪些上下文。工具返回的结果模型要看得见, 但不作为 assistant 的监督目标; 本例 172 个 token 里只有 75 个算 loss。'),
 ('rft','RFT: 当前策略采样 → 验证筛选 → SFT','llm_finetune.run_finetune.rft.train_rft',
  [('sft','示范 SFT'),('rft','拒绝采样微调'),('rl','在线 RL')],
  ['agent-trajectory-learning'],'采出来的候选全失败了, RFT 这一轮怎么更新?','这一轮不更新, 如实报告保留 0 条; 不能偷偷把标准答案塞进去充数。保留下来的轨迹还要看分布: 本例 256 条候选只留 18 条, 覆盖 16 个训练任务里的 10 个, 有的任务一条都没有。'),
 ('stream','流式消息: 什么时候才能执行工具','llm_agent.m15_claude_api.stream_demo',
  [('valid','完整流'),('disconnect','断流'),('malformed','破损 JSON')],
  ['agent-async-runtime'],'第一个工具的参数已经拼完整了, 能马上执行吗?','不能。要等到 message_stop、所有内容块、停止原因都收齐, 再过权限门才执行; 断流和破损 JSON 两组可交给权限门的工具数都是 0。增量文本可以边收边显示, 工具副作用必须等完整动作。'),
 ('trace','一次任务里的并行、重试与取消','llm_agent.m31_eval_suite.trace_demo',
  [('parallel','并行工具'),('retry','限流重试'),('cancel','取消任务')],
  ['agent-eval-suite'],'为什么 3 个模型 attempt 不等于 3 次逻辑请求?','因为其中 1 次是同一个逻辑请求被限流后的重试: 限流重试那组 attempt 3 次, 逻辑请求还是 2 次。计数要看 operation 和父 span, 不能数 attempt; 工具副作用也不能跟着重试再来一遍。'),
 ('retrieval','语义表示、索引与重排分别错在哪里','llm_agent.m45_semantic_retrieval.demo',
  [('exact','精确向量'),('ann','分区近似'),('rerank','top-4 重排'),('wide','全量重排'),('hybrid','编号精确约束')],
  ['agent-vector-index'],'重排的正确率低, 先看哪个中间量?','先看候选召回。正确文档不在候选里, 重排器再强也选不中: top-4 重排 4 题对 2, 放开到全量重排就对 3。再把编码器的丢失和 ANN 分区的丢失分开看。'),
 ('parallel','DP×TP×PP: 显存和时间一起算','llm_train.m22_parallel_budget.demo',
  [('balanced','DP2 TP2 PP2'),('more-tp','DP1 TP4 PP2'),('no-overlap','不重叠 DP'),('high-latency','高通信延迟'),('invalid','非法配置')],
  ['train-model-parallel'],'每卡参数占用减半了, 训练一定更快吗?','不一定。TP 从 2 提到 4, 每卡状态从 3.725 GiB 降到 1.863 GiB, 估算步时间却从 707.99 ms 涨到 975.24 ms: 更多 TP 小消息增加了启动延迟和同步等待。要把状态、激活、计算/访存、通信和气泡放在一起算。'),
 ('capacity','到达率上去了, 队列里会发生什么','llm_infer.m28_service_capacity.demo',
  [(f'{r}-{p}',f'{r}/s · '+('有界' if p=='bounded' else '无界')) for r in (8,24,48,96) for p in ('bounded','unbounded')],
  ['infer-multi-replica-routing'],'多拒绝一些请求换来更低的 p99, 能直接说服务变好了吗?','不能。分位数只统计完成了的请求: 48/s 有界队列拒掉 131 条, TTFT p99 从无界的 955.49 ms 降到 368.35 ms, 但这是把人挡在门外换来的。到达量、拒绝、超时、goodput 和资源回收要一起报。'),
]

def digest():
    h=hashlib.sha256()
    files=sorted(p for pkg in ('llm_agent','llm_finetune','llm_models','llm_train','llm_infer') for p in (ROOT/pkg).rglob('*.py'))+[Path(__file__).resolve()]
    for path in files:
        h.update(str(path.relative_to(ROOT)).encode());h.update(b'\0');h.update(path.read_bytes());h.update(b'\0')
    return h.hexdigest()

def export():
    entries=[]
    for key,title,module,conditions,routes,ask,answer in CASES:
        experiment=importlib.import_module(module).experiment
        entries.append({'id':key,'title':title,'module':module.replace('.','/')+'.py',
                        'run':'python -m '+module,'routes':routes,'challenge':{'ask':ask,'answer':answer},
                        'conditions':[{'id':mode,'label':label,**experiment(mode)} for mode,label in conditions]})
    result={'source_digest':digest(),'kind':'Python 执行记录回放；切换条件读取预先运行的结果，不在浏览器训练或调用 API。','entries':entries}
    encoded=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if '--check' in sys.argv:
        assert OUT.read_text()==encoded,'源码或执行结果已变化，请重新导出'
        print(f'7 组 / {sum(len(e["conditions"]) for e in entries)} 个条件的源码与执行记录一致')
    else:OUT.write_text(encoded);print(OUT.relative_to(ROOT))

if __name__=='__main__':export()
