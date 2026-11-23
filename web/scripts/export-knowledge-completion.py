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
  ['finetune-sft'], "多轮对话监督微调中同时存在两种 mask:\n- loss mask 用于标记哪些位置的预测误差计入训练损失。\n- attention mask 用于控制模型在前向计算时允许关注哪些上下文位置。\n训练时能否直接将 loss mask 当作 attention mask 使用?","不能混用, 两种 mask 承担完全不同的职责:\n- loss mask 决定监督信号的作用位置: 仅 assistant 回复内容及其结束标记计入训练损失, 其余位置的 labels 均设为 -100 以排除在损失计算之外。\n- attention mask 决定前向计算的上下文可见性: 除因果约束外, 它仅屏蔽为对齐 batch 序列长度而填充的 padding 位置。\n用户输入与工具返回结果虽然不计入 loss, 但模型在生成回复时必须通过注意力读取这些前文信息; 若将 loss mask 直接用作 attention mask, 用户提问与工具结果都会被屏蔽, 模型将在缺失上下文的条件下强行拟合回复。本例包含 7 条消息 (含一次工具调用及返回结果), 编码后共 172 个 token, 其中仅 75 个位置计入 loss (含 2 个受监督的 EOS 结束标记)。执行记录中的 labels 已完成左移对齐: 位置 t 的标签对应第 t+1 个 token, 即每个位置负责预测下一个 token。\n此外, 实验还验证了两类非法输入门禁:\n- 将工具返回结果伪装为 user 角色消息。\n- 将最大长度上限设为 40 以触发截断半条消息。\n这两种异常输入均在预处理阶段被直接拒绝, 「输入通过校验」读数为否。"),
 ('rft','RFT: 当前策略采样 → 验证筛选 → SFT','llm_finetune.run_finetune.rft.train_rft',
  [('sft','示范 SFT'),('rft','拒绝采样微调'),('rl','在线 RL')],
  ['agent-trajectory-learning'],"在拒绝采样微调 (RFT, 即对当前策略采样候选轨迹、经独立验证器筛选通过后再执行 SFT) 中, 若某一轮采样出的全部候选均未通过验证, 应当如何处理参数更新?","本轮应跳过参数更新, 并如实记录保留轨迹数为 0。RFT 的标准执行流程分为三步:\n1. 使用当前策略对每个训练任务各采样 16 次, 共生成 256 条候选轨迹。\n2. 调用独立的执行验证器逐条检验轨迹是否通过。\n3. 仅使用通过验证的轨迹执行标准监督微调。\n当通过验证的轨迹数为 0 时, 训练循环直接跳过本轮梯度更新并保持模型参数不变: 既不能在空样本集上计算损失, 也不能混入人工编写的标准示范充当采样结果, 否则算法语义便退化为示范 SFT。即便有轨迹通过验证, 还需检查任务覆盖分布: 本例正常条件下 256 条候选仅保留 18 条, 覆盖 16 个训练任务中的 10 个, 其余 6 个任务保留 0 条因而未获得直接监督信号。在玩具规模对照中, 示范 SFT、RFT 与在线 RL 三个条件的留出集通过率均从 0 升至 1, RFT 并未表现出优于示范 SFT 与在线 RL 的最终指标。"),
 ('stream','流式消息: 什么时候才能执行工具','llm_agent.m15_claude_api.stream_demo',
  [('valid','完整流'),('disconnect','断流'),('malformed','破损 JSON')],
  ['agent-async-runtime'],"模型以流式增量返回一条包含两个工具调用的响应消息时, 若第一个工具调用的参数 JSON 已接收完整, 能否立即触发该工具执行?","不能立即执行。执行器必须等待整条流式消息完整收束:\n- 收到标记消息结束的 message_stop 事件。\n- 消息内的每一个内容块均已正常闭合。\n- 服务端明确返回合法的停止原因。\n唯有上述条件全部满足后, 才能将解析出的工具调用移交权限门禁做授权校验。之所以不能提前执行, 是因为后续流式片段仍可能出现异常:\n- 第二个工具调用的参数 JSON 可能截断或格式损坏。\n- 网络流可能在中途异常断开。\n- 最终停止原因可能指示整条响应生成失败。\n本实验中, 「断流」组缺失末尾的 message_stop 事件, 「破损 JSON」组包含无法解析的工具参数片段, 两组可移交权限门禁的工具调用数均为 0, 且已开始传输的响应流不会自动静默重发。纯文本内容可以边接收边增量渲染, 但带有外部副作用的工具调用必须等待整条消息校验完整后方可执行。此外, 「完整流」组的服务端输出 token 读数为 6: 流式事件先后上报的 1、4、6 为累计计数, 应取末次读数 6, 而非累加为 11。"),
 ('trace','一次任务里的并行、重试与取消','llm_agent.m31_eval_suite.trace_demo',
  [('parallel','并行工具'),('retry','限流重试'),('cancel','取消任务')],
  ['agent-eval-suite'],"在一次 agent 任务的分布式追踪 trace 中, 模型调用 attempt 共记录了 3 次, 为什么对应的逻辑模型请求却只有 2 次?","因为其中 1 次 attempt 是同一逻辑请求遭遇服务端限流后的网络重试。逻辑请求代表 agent 在状态机层面向模型发起的一次推理步, 而 attempt 代表底层适配器实际发出的一次 HTTP/API 调用。本例中 agent 完成任务共需 2 次逻辑模型请求:\n1. 第一次请求模型并获得两个并行工具调用指令。\n2. 两个工具并行执行完毕后发起第二次请求, 汇总工具结果生成最终回复。\n在「限流重试」组中, 第一次逻辑请求的首个 API 调用因服务端返回状态码 429 被拒绝, 适配器在同一逻辑请求下自动重发一次, 因此底层 attempt 增至 3 次, 顶层逻辑请求仍为 2 次, 总 span 数量也相应从 7 增至 8。由此对应两条工程约束:\n- 统计业务步数与请求量时必须依据 operation 类型及父子 span 层级聚合, 不能直接累加底层 attempt。\n- 限流重试仅针对失败的模型 API 调用, 不得连带重复执行具有副作用的工具节点, 实验三组的实际工具执行数均保持为 2。"),
 ('retrieval','语义表示、索引与重排分别错在哪里','llm_agent.m45_semantic_retrieval.demo',
  [('exact','精确向量'),('ann','分区近似'),('rerank','top-4 重排'),('wide','全量重排'),('hybrid','编号精确约束')],
  ['agent-vector-index'],"两阶段检索流水线依次执行以下两步:\n1. 通过向量相似度检索召回 top-4 候选文档。\n2. 使用交叉编码重排器对查询与候选逐一联合打分并选出首位文档。\n当重排后的最终准确率偏低时, 排查故障应优先检查哪个中间指标?","应优先检查第一阶段候选召回率, 即目标文档是否进入了 top-4 候选集。重排器的作用范围仅限于已召回的候选子集, 若目标文档在初检阶段漏召回, 重排器无论打分多准都无法将其排至首位。本实验的 4 道测试查询中, 「否定」查询 (不要 开启 邮箱) 对应的目标文档 disable email 未能进入向量初检的 top-4, 导致基于 top-4 候选的重排仅答对 2 题; 若改为对全部 8 条文档做全量重排, 该查询即可命中, 正确题数升至 3 题。确认候选召回覆盖后, 再按链路逐段拆解初检漏召回的成因:\n- 若精确向量检索 (exact) 本身未能召回目标文档, 说明瓶颈在于语义编码器表示能力或训练语料覆盖不足。\n- 若 exact 能够召回而分区近似检索 (ann) 漏召回, 则说明误差源于近似索引仅搜索了部分分区。\n此外, 「罕见编号」查询需启用词面精确匹配混合检索 (hybrid) 才能命中目标文档, 因为编号 ZX91 与 ZX92 均不在编码器词表内, 两条文档生成的稠密向量完全相同。"),
 ('parallel','DP×TP×PP: 显存和时间一起算','llm_train.m22_parallel_budget.demo',
  [('balanced','DP2 TP2 PP2'),('more-tp','DP1 TP4 PP2'),('no-overlap','不重叠 DP'),('high-latency','高通信延迟'),('invalid','非法配置')],
  ['train-model-parallel'],"8 卡训练配置从 DP2 TP2 PP2 调整为 DP1 TP4 PP2 后, 张量并行度 TP 从 2 增至 4, 单卡持久化的训练状态显存随之减半。这是否意味着单步训练耗时一定会缩短?","不一定, 在本实验的代价模型下反而更慢。尽管单卡训练状态显存从 3.725 GiB 降至 1.863 GiB, 估算的单步耗时却从 707.99 ms 增至 975.24 ms。性能回退的主因在于同步通信开销: TP 将层内矩阵运算切分到多张卡上, 每层前向与反向传播都必须执行跨卡集合通信以汇总激活或梯度, 后续层依赖该通信结果因而难以将其重叠隐藏在计算背后。本次配置切换同时放大了两项通信成本:\n- TP 度从 2 增至 4, 单次集合通信跨越的设备数增加。\n- DP 度从 2 降至 1, 单个流水线副本需要串行处理的 micro-batch 数量从 8 个翻倍至 16 个, 跨卡同步次数随之翻倍。\n执行记录显示, TP 通信耗时从 134.99 ms 增至 423.41 ms, 关键路径设备上的通信调用次数从 786 次增至 4640 次; 若再将单次通信启动延迟从 8 μs 上调至 100 μs (「高通信延迟」组), 单步耗时进一步增至 1428.8 ms。因此评估并行策略时必须将训练状态显存、激活显存、计算与访存耗时、集合通信开销以及 PP 流水线气泡放在同一代价模型下联合权衡。"),
 ('capacity','到达率上去了, 队列里会发生什么','llm_infer.m28_service_capacity.demo',
  [(f'{r}-{p}',f'{r}/s · '+('有界' if p=='bounded' else '无界')) for r in (8,24,48,96) for p in ('bounded','unbounded')],
  ['infer-multi-replica-routing'],"当请求到达率升至 48/s 时, 容量上限为 8 的有界队列相比无界队列主动拒绝了大量过载请求, 同时测得更低的 TTFT p99。仅凭 TTFT p99 下降能否直接断定服务质量变好?","不能仅凭分位数延迟直接下结论。TTFT p99 等分位数指标通常只统计成功完成的请求样本, 被拒绝、排队超时或中途取消的请求均未纳入分母, 存在明显的幸存者偏差。在 48/s 到达率下, 有界队列主动拒绝了 131 条请求, 才使完成请求的 TTFT p99 从无界队列的 955.49 ms 降至 368.35 ms, 部分延迟下降源自准入丢弃。完整评估必须同时汇报总到达量、完成数、拒绝数、超时数、goodput 与异常终态资源回收情况。本实验定义的 goodput 指每秒同时满足以下三项服务等级目标 (SLO) 的请求吞吐:\n- TTFT 不超过 200 ms。\n- 相邻输出 token 间隔不超过 6 ms。\n- 端到端总耗时在 1 秒以内完成。\n结合全部请求终态与 goodput 后才能确认有界队列在过载下确实更优:\n- 有界队列的 goodput 达到每秒 16.51 条, 而无界队列仅为每秒 1.55 条。\n- 无界队列中有 93 条请求因长时间排队而超时失败, 即便最终完成的请求其 TTFT p95 也高达 943.98 ms, 绝大部分已违背 200 ms 的首 token 目标。\n因此服务质量改善的依据应当是 goodput 与全量终态分布, 而非孤立的完成请求 p99。"),
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
