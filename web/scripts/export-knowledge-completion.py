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
  ['finetune-sft'], '训练对话数据时有两种 mask: loss mask 标出哪些 token 要算进训练误差, attention mask 标出模型计算时能看到哪些 token。能不能把 loss mask 直接拿来当 attention mask 用?','不能, 两个 mask 管的是两件不同的事。loss mask 决定哪些位置当训练目标: 只有 assistant 的回复和它的结束标记算 loss, 其它位置的 labels 写成 -100, 表示不计入误差。attention mask 决定模型在每个位置能看到哪些上下文, 它只遮掉为了凑齐 batch 长度补上的 padding。用户的问题和工具返回的结果不算 loss, 但模型必须看得见它们, 不然就成了没看题目就学写答案; 拿 loss mask 当 attention mask, 这些上下文就被遮掉了。本例 7 轮对话 (含一次工具调用和它的结果) 共 172 个 token, 只有 75 个算 loss, 其中包括 2 个被监督的 EOS (回复结束标记)。执行记录里的 labels 已经左移一位: 第 t 个位置的 label 是第 t+1 个 token, 因为模型在每个位置要预测的是下一个 token。另外两个条件演示输入校验: 把工具结果伪装成 user 消息, 或者把长度上限设成 40 逼它截掉半条消息, 都会被直接拒绝, 「输入通过校验」为否。'),
 ('rft','RFT: 当前策略采样 → 验证筛选 → SFT','llm_finetune.run_finetune.rft.train_rft',
  [('sft','示范 SFT'),('rft','拒绝采样微调'),('rl','在线 RL')],
  ['agent-trajectory-learning'],'RFT (拒绝采样微调: 让模型自己做题, 只把做对的答案拿来当训练样本) 这一轮采出来的候选全失败了, 该怎么更新?','这一轮不更新, 如实报告保留了 0 条。RFT 的流程是: 让当前模型对每个训练任务各尝试 16 次, 共 256 条轨迹 (一条轨迹是模型完成一个任务的整串动作), 用独立的执行验证器检查每条是否合格, 只拿合格的做普通的监督训练。一条合格的都没有时, 代码直接跳过更新, 参数不变; 不能对空数据算 loss, 更不能偷偷把人写的标准答案塞进去充数, 那样实际做的就成了示范 SFT (用人写的正确示范来训练)。保留下来的轨迹还要看分布: 本例 256 条只留下 18 条, 覆盖 16 个训练任务里的 10 个, 另外 6 个任务一条都没有, 没被直接教到。三个条件的留出通过率 (在训练时没见过的任务上的通过率) 都从 0 升到 1, RFT 并没有胜过示范 SFT 和在线 RL。'),
 ('stream','流式消息: 什么时候才能执行工具','llm_agent.m15_claude_api.stream_demo',
  [('valid','完整流'),('disconnect','断流'),('malformed','破损 JSON')],
  ['agent-async-runtime'],'模型以流式 (一小块一小块地) 返回一条要调用两个工具的消息。第一个工具的参数 JSON 已经收完整了, 能马上执行它吗?','不能。要等整条消息结束: 收到表示结束的 message_stop 事件, 每个内容块都已闭合, 停止原因也到了, 再把工具调用交给权限门 (执行前判断这个工具能不能调用的一层)。因为后面还可能出问题: 第二个工具的参数可能残缺, 流可能中途断掉, 停止原因也可能表明这条消息失败了。本例「断流」组少了最后的 message_stop, 「破损 JSON」组把一段参数改成了无法解析的内容, 两组可交给权限门的工具数都是 0, 已经开始的流也不会自动重发。文字可以边收边显示, 但工具的副作用 (例如写文件、调接口) 必须等动作完整才能执行。另外, 「完整流」组的服务端输出 token 是 6: 流里先后报了 1、4、6, 这是累计值, 取最后一个, 不能加成 11。'),
 ('trace','一次任务里的并行、重试与取消','llm_agent.m31_eval_suite.trace_demo',
  [('parallel','并行工具'),('retry','限流重试'),('cancel','取消任务')],
  ['agent-eval-suite'],'一次 agent 任务的 trace (把每一步按父子关系记成一棵树, 每个节点叫 span) 里, 模型 attempt (实际发出的 API 调用) 有 3 次, 为什么逻辑请求只有 2 次?','因为其中 1 次是同一个逻辑请求被限流后的重试。逻辑请求是 agent 想问模型的一次问题, attempt 是为它实际发出的一次 API 调用。本例 agent 一共要问模型两次: 第一次拿到两个工具调用, 两个工具并行跑完后再问第二次, 拿到最终回答。「限流重试」组的第一次 API 调用被服务端以 429 (请求太频繁) 拒绝, 适配器再发一次, 于是 attempt 有 3 次, 逻辑请求仍是 2 次, span 数也从 7 变成 8。所以计数要看 operation 和父 span, 不能直接数 attempt; 而且重试的只是模型请求, 工具的副作用不能跟着再执行一遍, 三组的工具执行数都是 2。'),
 ('retrieval','语义表示、索引与重排分别错在哪里','llm_agent.m45_semantic_retrieval.demo',
  [('exact','精确向量'),('ann','分区近似'),('rerank','top-4 重排'),('wide','全量重排'),('hybrid','编号精确约束')],
  ['agent-vector-index'],'检索分两步: 先用向量相似度找出 top-4 候选, 再用重排器 (把问题和每个候选放在一起打分的小模型) 从中挑一个。重排后的正确率低, 应该先看哪个中间量?','先看候选召回: 正确文档有没有进入 top-4 候选。重排器只能在候选里挑, 正确文档不在里面, 它再强也选不中。本例 4 道测试题里, 「否定」那题 (不要 开启 邮箱) 的正确答案 disable email 没进 top-4, 所以 top-4 重排只对 2 题; 放开到全量重排 (对全部 8 条文档打分), 这题就对了, 一共对 3 题。召回没问题之后, 再把丢答案的原因分开看: 精确向量检索 (exact) 就排错, 是编码器 (把文本变成向量的模型) 或训练覆盖的问题; exact 对而分区近似 (ann) 错, 是近似索引只搜了一部分分区造成的。「罕见编号」那题要靠编号精确匹配 (hybrid) 才答对, 因为 ZX91 和 ZX92 都不在词表里, 两条文档的向量完全一样。'),
 ('parallel','DP×TP×PP: 显存和时间一起算','llm_train.m22_parallel_budget.demo',
  [('balanced','DP2 TP2 PP2'),('more-tp','DP1 TP4 PP2'),('no-overlap','不重叠 DP'),('high-latency','高通信延迟'),('invalid','非法配置')],
  ['train-model-parallel'],'8 张卡从 DP2 TP2 PP2 换成 DP1 TP4 PP2: 张量并行 TP (把每一层的矩阵切开, 分给几张卡一起算) 从 2 提到 4, 每张卡要存的训练状态减半。训练一定更快吗?','不一定, 本例反而更慢。每卡训练状态从 3.725 GiB 降到 1.863 GiB, 估算的单步时间却从 707.99 ms 涨到 975.24 ms。原因在通信: TP 把一层的计算切到几张卡上, 每层的前向和反向都要在这几张卡之间交换中间结果, 下一层要等交换完成才能算, 这段时间藏不进计算里。TP 从 2 变 4, 每次交换要在更多卡之间进行; DP (数据并行, 每组卡各算一部分 batch) 从 2 变 1, 每组要处理的微批 (micro-batch) 从 8 个变成 16 个, 交换次数又翻了一倍。执行记录里 TP 通信时间从 134.99 ms 涨到 423.41 ms, 右栏关键卡上的通信次数从 786 涨到 4640; 再把每条消息的启动延迟从 8 μs 调到 100 μs (「高通信延迟」组), 步时间涨到 1428.8 ms。所以要把训练状态、激活、计算和访存、通信、流水线气泡 (PP 各阶段互相等待的空闲时间) 放在一起算。'),
 ('capacity','到达率上去了, 队列里会发生什么','llm_infer.m28_service_capacity.demo',
  [(f'{r}-{p}',f'{r}/s · '+('有界' if p=='bounded' else '无界')) for r in (8,24,48,96) for p in ('bounded','unbounded')],
  ['infer-multi-replica-routing'],'到达率 48/s 时, 有界队列 (最多排 8 个, 满了就拒绝) 比无界队列多拒绝了很多请求, 换来更低的 TTFT p99 (99% 的请求在这个时间内收到第一个输出 token)。能直接说服务变好了吗?','不能直接下结论。p99 这类分位数只统计完成了的请求, 被拒绝、超时、取消的请求根本不进统计。48/s 时有界队列拒掉 131 条, TTFT p99 从无界的 955.49 ms 降到 368.35 ms, 其中有一部分改善是把请求挡在门外换来的。要把到达量、完成、拒绝、超时、goodput 和资源回收一起报; goodput 是每秒有多少请求同时满足三条服务目标 (SLO): 首 token 不超过 200 ms、相邻 token 间隔不超过 6 ms、1 秒内完成。本例放在一起看之后, 有界队列确实更好: 满足 SLO 的请求每秒 16.51 条, 无界只有 1.55 条; 无界那边有 93 条排到超时, 完成的请求 TTFT p95 也到了 943.98 ms, 大多超过 200 ms 的目标。但这个结论来自 goodput 和全部终态, 不是来自 p99。'),
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
    result={'source_digest':digest(),'kind':'Python 执行记录回放: 切换条件读取预先运行的结果, 不在浏览器训练或调用 API。','entries':entries}
    encoded=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if '--check' in sys.argv:
        assert OUT.read_text()==encoded,'源码或执行结果已变化，请重新导出'
        print(f'7 组 / {sum(len(e["conditions"]) for e in entries)} 个条件的源码与执行记录一致')
    else:OUT.write_text(encoded);print(OUT.relative_to(ROOT))

if __name__=='__main__':export()
