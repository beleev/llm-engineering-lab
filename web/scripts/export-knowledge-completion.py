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
 ('chat','多轮消息：谁的 token 在计算 loss','llm_finetune.run_finetune.chat_template.demo',
  [('correct','正确模板'),('wrong-role','工具角色错配'),('truncated','截断半条消息')],
  ['finetune-sft'], 'loss mask 为什么不能拿来当 attention mask？','前者决定哪些目标参与训练，后者决定模型可见哪些上下文。工具观察应可见，但不作为 assistant 监督目标。'),
 ('rft','RFT：当前策略采样 → 验证筛选 → SFT','llm_finetune.run_finetune.rft.train_rft',
  [('sft','示范 SFT'),('rft','拒绝采样微调'),('rl','在线 RL')],
  ['finetune-sft','agent-trajectory-learning','agent-agent-training'],'候选全失败时，RFT 应如何更新？','跳过更新并报告零保留；不能暗中注入标准答案。还要审计保留任务的分布。'),
 ('stream','流式消息：什么时候可以执行工具','llm_agent.m15_claude_api.stream_demo',
  [('valid','完整流'),('disconnect','断流'),('malformed','破损 JSON')],
  ['agent-runtime-reliability','agent-async-runtime','agent-full-loop'],'第一条工具参数已经拼完，能立刻执行吗？','不能。完整停止事件、所有内容块、停止原因和权限门都检查后再执行。增量文本可展示，工具副作用要等完整动作。'),
 ('trace','一次任务里的并行、重试与取消','llm_agent.m31_eval_suite.trace_demo',
  [('parallel','并行工具'),('retry','限流重试'),('cancel','取消任务')],
  ['agent-eval-suite','agent-evals'],'为什么三个模型 attempt 不等于三次逻辑请求？','其中一次是同一逻辑请求的重试。计数依赖 operation 与父 span，工具副作用不能随重试重复。'),
 ('retrieval','语义表示、索引与重排分别错在哪里','llm_agent.m45_semantic_retrieval.demo',
  [('exact','精确向量'),('ann','分区近似'),('rerank','top-4 重排'),('wide','全量重排'),('hybrid','编号精确约束')],
  ['agent-rag','agent-vector-index'],'重排正确率低，应该先看哪个中间量？','先看候选召回。正确文档不在候选中时，无论重排器多强都无法选中。再区分编码器与 ANN 的丢失。'),
 ('parallel','DP×TP×PP：显存与时间一起算','llm_train.m22_parallel_budget.demo',
  [('balanced','DP2 TP2 PP2'),('more-tp','DP1 TP4 PP2'),('no-overlap','不重叠 DP'),('high-latency','高通信延迟'),('invalid','非法配置')],
  ['train-model-parallel','train-collectives-loop'],'参数占用减半，训练一定更快吗？','不一定。更多 TP 小消息会增加启动延迟和同步等待。要同时核对状态、激活、计算/访存、通信及气泡。'),
 ('capacity','到达率上升后，排队会发生什么','llm_infer.m28_service_capacity.demo',
  [(f'{r}-{p}',f'{r}/s · '+('有界' if p=='bounded' else '无界')) for r in (8,24,48,96) for p in ('bounded','unbounded')],
  ['infer-scheduler','infer-multi-replica-routing','infer-engine'],'拒绝更多请求换来更低 p99，能直接宣布服务更好吗？','不能。分位数只来自完成请求，要一起报告到达量、拒绝、超时、goodput 和资源回收。'),
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
