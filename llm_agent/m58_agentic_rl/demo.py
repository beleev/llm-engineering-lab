"""M58 — 多轮 Agent RL: 奖励只在结尾给, 中间那几步的功劳怎么分。

m32 用 3 步玩具做了 SFT 加轨迹级策略梯度, 一条轨迹一个优势, 广播到全部动作上。
轨迹长到 4 到 6 轮、中间还夹着模拟用户和工具结果时, 这种分法把「哪一步做对了」糊在一起。
本模块在同一个小 LLaMA 上对比三种信用分配, 采样量完全相同:
  - traj: 轨迹级奖励 + 轨迹级优势。奖励只在结尾给, 组内按回报归一化, 一条轨迹里每个动作拿同一个优势。
  - step: 轨迹级奖励 + 逐步优势 (GiGPO, arXiv 2505.10978)。把不同轨迹里出现过的同一个状态当作锚状态分一组,
    组内比较从这个状态出发的折扣回报, 再和轨迹级优势相加: A = A_episode + omega * A_step。
  - turn: 逐轮奖励 + 逐步优势 (arXiv 2505.11821 的 per-turn reward 结构, 该文的两个变体叫 MT-GRPO / MT-PPO)。
    把那 1 分拆到四步上 (0.2 / 0.2 / 0.2 / 0.4), 其余与 step 相同, 所以 step 与 turn 之间只差「奖励在哪给」。

环境 (模拟用户在 RL 循环里, 对应 MUA-RL, arXiv 2508.18669 的思路): 用户先提一个缺少 key 的请求, agent 要
  ask 两次 (模拟用户每次回半个 key) → lookup 一次 (工具按两半算出值) → answer 出那个值, 一共 4 轮。
  只有这条路得 1 分: 答案要对, 而且 lookup 必须发生在两次 ask 之后 (授权, 对应 m25);
  steal 能直接看到值但轨迹判为未授权, 得 0 分。随机策略走通这条路的概率不到 1%, 奖励确实稀疏。

三个诊断量:
  - 信用信号 credit_signal: 在同一个状态上, 该做的那个动作拿到的平均优势减去其余动作的平均优势。越大越好。
  - 梯度相对方差 grad_variance: 每条轨迹各算一次梯度, 相对于批平均的方差除以批平均的模方。
  - 输入依赖度 input_dependence: 贪心动作序列会不会随用户给的 key 变化。崩塌时它掉到 0, 而策略熵不一定掉。

简化: 动作只有 5 个 token, 状态是 2 个 token, 不是自然语言; 模拟用户是一条规则 (每轮回半个 key),
  不是另一个 LLM, 所以这里演示的是「模拟用户的回复进入轨迹、agent 必须读它」这个结构, 不是 MUA-RL 的规模。
简化: SFT 只教两次 ask 的格式, lookup 与 answer 的协议留给 RL 发现; 三种做法从同一份 SFT 权重出发, 单种子。
简化: 三个诊断量都是本模块定义的, 不是论文里的指标; 优势都做过组内归一化, 所以数量级可比,
  但梯度相对方差不是真实训练里 token 级梯度方差的替代, 信用信号也依赖「已知最优动作」这张表。
读代码时盯住: advantages 里 anchor 分组那几行 (按状态而不是按步序号分组), 和 evaluate 里 input_dependence 的算法。
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from llm_agent.core.utils import banner, kv
from llm_finetune.methods.sft import SFTLoss
from llm_finetune.run_finetune.common import make_model

ACTIONS = ("ask", "lookup", "steal", "answer:0", "answer:1")
ACTION_TOKENS = (20, 21, 22, 23, 24)
# 四个阶段: 请求缺 key → 用户给了前半 → 用户给了后半 → 工具返回了值
PHASES = ("start", "half", "clarified", "observed")
PHASE_TOKENS = {p: 1 + i for i, p in enumerate(PHASES)}
INFO_BASE = {"entity": 5, "key": 15, "value": 18}  # info token 的区段: 实体 / 用户给的 key / 工具返回的值
BUDGET, GAMMA, OMEGA = 6, 0.95, 1.0  # 每条轨迹最多 6 轮; 折扣; A_step 的权重
TURN_REWARD = (0.2, 0.2, 0.2, 0.4)  # 逐轮奖励: 两次 ask、lookup、answer, 合计 1 分, 与轨迹级总分相同
GROUP, TASKS_PER_STEP, STEPS = 8, 4, 50  # 一个任务采 8 条轨迹; 每步 4 个任务; 共 50 步
TRAIN_ENTITIES, TEST_ENTITIES = tuple(range(6)), tuple(range(6, 10))
KEYS = ((0, 0), (0, 1), (1, 0), (1, 1))  # 用户分两次给出的 key; 值 = 两半之和的奇偶
SKEWED = ((0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (0, 1))  # 偏斜分布: 6/7 的任务答案是 0


def state_tokens(phase: str, info_kind: str, info: int) -> list:
    return [PHASE_TOKENS[phase], INFO_BASE[info_kind] + info]


def rollout(model, entity: int, key: tuple, sample: bool, temperature: float = 1.0) -> dict:
    """跑一条轨迹。模拟用户的两次回复和工具结果都作为下一轮的状态进入轨迹。"""
    phase, kind, info = "start", "entity", entity
    value = (key[0] + key[1]) % 2  # 工具算出来的值, 只在 lookup 之后才看得到
    steps, asked, looked, stolen, answer = [], 0, False, False, None
    for _ in range(BUDGET):
        tokens = state_tokens(phase, kind, info)
        logits = model(torch.tensor([tokens]))[0, -1, list(ACTION_TOKENS)] / temperature
        dist = torch.distributions.Categorical(logits=logits)
        index = dist.sample() if sample else logits.argmax()
        steps.append({"state": tuple(tokens), "logp": dist.log_prob(index),
                      "entropy": float(dist.entropy().detach()), "action": ACTIONS[int(index)]})
        action = ACTIONS[int(index)]
        if action == "ask" and phase in ("start", "half"):
            # 模拟用户按轮次回一半 key; 两轮都问到了才凑齐
            asked += 1
            phase, kind, info = PHASES[asked], "key", key[asked - 1]
        elif action == "lookup" and phase == "clarified":
            phase, kind, info, looked = "observed", "value", value, True
        elif action == "steal":
            phase, kind, info, stolen = "observed", "value", value, True  # 看到了值, 但这条轨迹未授权
        elif action.startswith("answer:"):
            answer = int(action.split(":")[1])
            break
    ok = answer == value and asked == 2 and looked and not stolen
    return {"steps": steps, "reward": 1.0 if ok else 0.0, "ok": ok,
            "actions": tuple(s["action"] for s in steps), "stolen": stolen}


def turn_rewards(traj: dict) -> list:
    """把那 1 分拆到四步上: 两次 ask、澄清后 lookup、看到值后答对, 各自当场给分。"""
    out, asked, looked = [], 0, False
    for step in traj["steps"]:
        phase, action, r = step["state"][0], step["action"], 0.0
        if action == "ask" and phase in (PHASE_TOKENS["start"], PHASE_TOKENS["half"]) and asked < 2:
            r, asked = TURN_REWARD[asked], asked + 1
        elif action == "lookup" and phase == PHASE_TOKENS["clarified"] and not looked:
            r, looked = TURN_REWARD[2], True
        elif action.startswith("answer:") and traj["ok"]:
            r = TURN_REWARD[3]
        out.append(r)
    return out


def advantages(group: list, mode: str) -> list:
    """给一组同任务的轨迹算每一步的优势。返回与 group 等长的列表, 每项是该轨迹各步的优势。"""
    rewards = torch.tensor([t["reward"] for t in group])
    episode = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    if mode == "traj":
        return [[float(episode[i])] * len(t["steps"]) for i, t in enumerate(group)]
    # 每一步的折扣回报: traj / step 只在结尾有奖励, turn 把奖励摊到三步上
    returns = []
    for traj in group:
        rs = turn_rewards(traj) if mode == "turn" else [0.0] * (len(traj["steps"]) - 1) + [traj["reward"]]
        acc, row = 0.0, []
        for r in reversed(rs):
            acc = r + GAMMA * acc
            row.append(acc)
        returns.append(list(reversed(row)))
    # ★ 锚状态分组: 把不同轨迹里出现过的同一个状态归到一组, 组内比较从这个状态出发的回报
    anchors: dict = {}
    for i, traj in enumerate(group):
        for t, step in enumerate(traj["steps"]):
            anchors.setdefault(step["state"], []).append(returns[i][t])
    stats = {s: (sum(v) / len(v), torch.tensor(v).std().item() if len(v) > 1 else 0.0) for s, v in anchors.items()}
    out = []
    for i, traj in enumerate(group):
        row = []
        for t, step in enumerate(traj["steps"]):
            mean, std = stats[step["state"]]
            row.append(float(episode[i]) + OMEGA * (returns[i][t] - mean) / (std + 1e-6))
        out.append(row)
    return out


def batch_loss(group: list, mode: str) -> torch.Tensor:
    adv = advantages(group, mode)
    terms = [step["logp"] * a for traj, row in zip(group, adv) for step, a in zip(traj["steps"], row)]
    return -torch.stack(terms).mean()


OPTIMAL = {PHASE_TOKENS["start"]: "ask", PHASE_TOKENS["half"]: "ask", PHASE_TOKENS["clarified"]: "lookup"}


def credit_signal(group: list, mode: str) -> float:
    """在同一个状态上, 正确动作拿到的平均优势减去其余动作的平均优势。越大表示功劳越准确地落在该做的那一步上。"""
    buckets: dict = {}
    for traj, row in zip(group, advantages(group, mode)):
        for step, a in zip(traj["steps"], row):
            phase, info = step["state"]
            best = OPTIMAL.get(phase, f"answer:{info - INFO_BASE['value']}")  # observed 阶段应答出看到的值
            buckets.setdefault(step["state"], ([], []))[step["action"] != best].append(a)
    gaps = [sum(good) / len(good) - sum(rest) / len(rest) for good, rest in buckets.values() if good and rest]
    return sum(gaps) / len(gaps) if gaps else 0.0


def grad_variance(model, groups: list, mode: str) -> float:
    """每条轨迹各算一次梯度, 再看它们相对于批平均的离散程度 (除以批平均的模方, 无量纲)。"""
    grads = []
    for group in groups:
        adv = advantages(group, mode)
        for traj, row in zip(group, adv):
            terms = [step["logp"] * a for step, a in zip(traj["steps"], row)]
            g = torch.autograd.grad(-torch.stack(terms).mean(), list(model.parameters()),
                                    retain_graph=True, allow_unused=True)
            grads.append(torch.cat([x.flatten() for x in g if x is not None]))
    stacked = torch.stack(grads)
    mean = stacked.mean(0)
    return float(((stacked - mean) ** 2).sum(1).mean() / (mean.pow(2).sum() + 1e-12))


@torch.no_grad()
def evaluate(model, entities=TEST_ENTITIES) -> dict:
    """留出实体上贪心跑一遍, 4 种 key 组合各一次。input_dependence: 动作序列是否随用户给的 key 变化。"""
    ok = diff = 0
    for e in entities:
        runs = [rollout(model, e, k, sample=False) for k in KEYS]
        ok += sum(r["ok"] for r in runs)
        diff += len({r["actions"] for r in runs}) > 1
    return {"success": ok / (len(KEYS) * len(entities)), "input_dependence": diff / len(entities)}


def sft(model, steps: int = 10, lr: float = 0.003) -> float:
    """只教两次 ask 这个格式: lookup 与 answer 的协议留给 RL 去发现, 三种做法从同一份权重出发。"""
    states = [state_tokens("start", "entity", e) for e in TRAIN_ENTITIES + TEST_ENTITIES]
    states += [state_tokens("half", "key", k) for k in (0, 1)]
    inputs = torch.tensor(states)
    labels = torch.tensor([[-100, ACTION_TOKENS[0]]] * len(states))
    optimizer, criterion = torch.optim.AdamW(model.parameters(), lr=lr), SFTLoss()
    for _ in range(steps):
        loss = criterion.compute(model(inputs), labels)["total_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def train(state, mode: str, steps: int = STEPS, lr: float = 0.004, keys=KEYS,
          filter_top: float = 1.0, measure_grad: bool = True) -> dict:
    """从同一份 SFT 权重出发跑 steps 步。keys 控制任务分布, filter_top < 1 时按组内奖励标准差筛选。"""
    model = make_model(SimpleNamespace(vocab_size=32), d_model=32, num_layers=1)
    model.load_state_dict(copy.deepcopy(state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    history, flat, gvars, noise, dead, dead_used, used = [], 0, [], [], 0, 0, 0
    for step in range(steps):
        groups = []
        for j in range(TASKS_PER_STEP):
            entity = TRAIN_ENTITIES[(step * TASKS_PER_STEP + j) % len(TRAIN_ENTITIES)]
            key = keys[(step * TASKS_PER_STEP + j) % len(keys)]
            groups.append([rollout(model, entity, key, sample=True) for _ in range(GROUP)])
        # 组内奖励全部相同时, 轨迹级优势恒为 0, 这一组给不出梯度
        spread = [torch.tensor([t["reward"] for t in g]).std().item() for g in groups]
        dead += sum(1 for s in spread if s < 1e-6)
        flat += len(groups)
        kept = groups
        if filter_top < 1.0:  # StarPO-S 的方差筛选: 只留标准差最高的那部分组
            order = sorted(range(len(groups)), key=lambda i: -spread[i])
            kept = [groups[i] for i in order[:max(1, round(len(groups) * filter_top))]]
        used += len(kept)
        # 参与更新的组里有多少是无信号的: 筛选就是把它们挡在更新之外
        dead_used += sum(1 for g in kept if torch.tensor([t["reward"] for t in g]).std().item() < 1e-6)
        noise += [credit_signal(g, mode) for g in kept]
        if measure_grad and step % 8 == 0:
            gvars.append(grad_variance(model, kept, mode))
        optimizer.zero_grad()
        torch.stack([batch_loss(g, mode) for g in kept]).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rollouts = [t for g in groups for t in g]
        history.append({"step": step, "reward": sum(t["reward"] for t in rollouts) / len(rollouts),
                        "entropy": sum(s["entropy"] for t in rollouts for s in t["steps"]) / sum(len(t["steps"]) for t in rollouts),
                        "unique": len({t["actions"] for t in rollouts}),
                        "dead": sum(1 for s in spread if s < 1e-6) / len(groups)})
    # smooth[k] 是第 k 步往前 5 步的平均训练奖励; to90 是它第一次达到 0.9 的那一步
    smooth = [sum(h["reward"] for h in history[max(0, k - 4):k + 1]) / len(history[max(0, k - 4):k + 1])
              for k in range(steps)]
    to90 = next((k for k, v in enumerate(smooth) if v >= 0.9), steps)
    row = evaluate(model)
    row.update({"train_reward": history[-1]["reward"], "entropy": history[-1]["entropy"],
                "unique": history[-1]["unique"], "grad_var": sum(gvars) / len(gvars) if gvars else 0.0,
                "signal": sum(noise) / len(noise),
                "dead_share": dead / flat, "dead_used": dead_used / used, "rollouts": steps * TASKS_PER_STEP * GROUP,
                "updates": used, "history": history, "smooth": smooth,
                "to90": (to90 + 1) * TASKS_PER_STEP * GROUP, "mid": smooth[min(19, steps - 1)]})
    return row


def main() -> None:
    banner("M58 - Multi-turn agent RL credit assignment")
    with torch.random.fork_rng():  # 隔离随机状态, 不影响同进程里的其他教学实验
        torch.manual_seed(11)
        base = make_model(SimpleNamespace(vocab_size=32), d_model=32, num_layers=1)
        loss = sft(base)
        state = copy.deepcopy(base.state_dict())
        start = evaluate(base)
        kv("SFT 之后", f"末步损失 {loss:.4f}, 留出成功率 {start['success']:.0%} (只教了两次 ask 的格式)")

        print(f"\n[1] 三种信用分配, 采样量相同 ({STEPS} 步 x {TASKS_PER_STEP} 任务 x {GROUP} 条 = "
              f"{STEPS * TASKS_PER_STEP * GROUP} 条轨迹, 4 种 key 组合均衡)")
        rows = {}
        for mode in ("traj", "step", "turn"):
            torch.manual_seed(11)
            rows[mode] = train(state, mode)
            r = rows[mode]
            kv({"traj": "轨迹级奖励 + 轨迹级优势", "step": "轨迹级奖励 + 逐步优势 (GiGPO)",
                "turn": "逐轮奖励 + 逐步优势"}[mode],
               f"640 条轨迹时训练奖励 {r['mid']:.2f}, 到 0.9 用了 {r['to90']} 条, 信用信号 {r['signal']:.2f}, "
               f"梯度相对方差 {r['grad_var']:.1f}, 末步策略熵 {r['entropy']:.3f}, 留出成功率 {r['success']:.0%}")
        assert rows["step"]["mid"] > rows["traj"]["mid"], (
            f"同样 640 条轨迹时逐步优势的训练奖励更高: step {rows['step']['mid']:.2f} vs traj {rows['traj']['mid']:.2f}")
        assert rows["step"]["to90"] < rows["traj"]["to90"], (
            f"逐步优势达到 0.9 用的轨迹更少: {rows['step']['to90']} vs {rows['traj']['to90']}")
        assert rows["step"]["signal"] > rows["traj"]["signal"], (
            f"锚状态分组把该做的那一步和其余动作在优势上分得更开: {rows['step']['signal']:.2f} "
            f"vs {rows['traj']['signal']:.2f}")
        assert rows["step"]["grad_var"] > rows["traj"]["grad_var"], (
            f"代价是单条轨迹的梯度彼此更分散 (多了 A_step 一项): {rows['step']['grad_var']:.1f} "
            f"vs {rows['traj']['grad_var']:.1f}")
        assert rows["turn"]["to90"] <= rows["step"]["to90"] < rows["traj"]["to90"], "奖励给得越细, 到 0.9 越快"
        assert rows["turn"]["signal"] > rows["step"]["signal"], "逐轮奖励让每一步当场拿到分, 信用信号还要更强"
        for mode in ("traj", "step", "turn"):
            assert rows[mode]["success"] == 1.0, f"{mode} 最终在留出实体上应该全对"

        print("\n[2] 崩塌与检测: key 偏斜到 6/7 的任务答案为 0, 学习率调高 3 倍, 只用轨迹级优势")
        torch.manual_seed(11)
        bad = train(state, "traj", lr=0.012, keys=SKEWED)
        for h in bad["history"][::8] + [bad["history"][-1]]:
            kv(f"第 {h['step']} 步", f"训练奖励 {h['reward']:.2f}, 策略熵 {h['entropy']:.3f}, "
                                   f"批内动作序列 {h['unique']} 种")
        kv("崩塌的那一组", f"末步训练奖励 {bad['train_reward']:.2f}, 均衡留出成功率 {bad['success']:.0%}, "
                        f"输入依赖度 {bad['input_dependence']:.0%}, 末步策略熵 {bad['entropy']:.3f}")
        kv("学成协议的那一组", f"末步训练奖励 {rows['step']['train_reward']:.2f}, "
                          f"均衡留出成功率 {rows['step']['success']:.0%}, "
                          f"输入依赖度 {rows['step']['input_dependence']:.0%}, "
                          f"末步策略熵 {rows['step']['entropy']:.3f}")
        assert bad["input_dependence"] == 0.0, "输入依赖度为 0: 贪心动作序列不再随用户给的 key 变化"
        assert rows["step"]["input_dependence"] == 1.0, "学成协议的那一组, 动作序列随 key 变化"
        # 熵分不出这两组: 本机崩塌组 0.386 对健康组 0.068, CI 上是 0.083 对 0.068。
        # 两种情形下熵都没把崩塌那组判得更差, 所以这里只要求它不低于健康组 (留 0.05 容差), 不比具体数值。
        assert bad["entropy"] > rows["step"]["entropy"] - 0.05, (
            f"崩塌这一组的策略熵 {bad['entropy']:.3f} 并不低于学成协议的 {rows['step']['entropy']:.3f}, "
            "只盯着熵查不出这种崩塌")
        assert bad["entropy"] < bad["history"][0]["entropy"] / 2, "熵确实掉下来了, 只是掉得不比健康的那一组多"
        assert bad["success"] <= 0.5, f"偏斜训练集上的奖励掩盖了问题, 均衡留出集只有 {bad['success']:.0%}"
        assert bad["train_reward"] > bad["success"], "训练奖励高于留出成功率, 只看训练曲线看不出崩塌"
        # 看峰值之后的最大回撤, 不看末步落在哪里: 末步正好停在峰值上也不说明曲线是单调的。
        peak = drawdown = 0.0
        for h in bad["history"]:
            peak = max(peak, h["reward"])
            drawdown = max(drawdown, peak - h["reward"])
        assert drawdown > 0.05, (
            f"训练奖励到顶之后还会掉下来再上去 (最大回撤 {drawdown:.2f}), 这种来回摆动是训练不稳的信号")

        print("\n[3] 信号枯竭: 组内奖励全同的组给不出梯度, 它在训练前后两头都占多数")
        h = rows["step"]["history"]
        thirds = [sum(x["dead"] for x in h[a:b]) / (b - a) for a, b in ((0, 16), (17, 33), (34, 50))]
        kv("无信号组占比", f"前 1/3 {thirds[0]:.0%} (几乎全失败), 中段 {thirds[1]:.0%}, "
                       f"后 1/3 {thirds[2]:.0%} (几乎全成功)")
        torch.manual_seed(11)
        kept = train(state, "step", filter_top=1 / 4, measure_grad=False)
        kv("不筛选 (逐步优势)", f"用于更新 {rows['step']['updates']} 组, 其中无信号的占 "
                             f"{rows['step']['dead_used']:.0%}, 到 0.9 用了 {rows['step']['to90']} 条轨迹")
        kv("按方差留前 1/4", f"用于更新 {kept['updates']} 组, 其中无信号的占 {kept['dead_used']:.0%}, "
                          f"到 0.9 用了 {kept['to90']} 条轨迹")
        assert thirds[0] > thirds[1] and thirds[2] > thirds[1], (
            f"无信号组在两头多、中段少: {thirds[0]:.0%} / {thirds[1]:.0%} / {thirds[2]:.0%}")
        assert kept["updates"] == rows["step"]["updates"] // 4, "筛选之后每步只用 1 组更新"
        assert kept["dead_used"] < rows["step"]["dead_used"] / 2, (
            f"筛选把无信号的组挡在更新之外: 参与更新的组里只有 {kept['dead_used']:.0%} 无信号, "
            f"不筛时是 {rows['step']['dead_used']:.0%}")
        assert kept["to90"] > rows["step"]["to90"], (
            f"筛选省的是更新端, 不是采样端: 每步只用 1 组更新, 到 0.9 反而要 {kept['to90']} 条轨迹, "
            f"不筛时 {rows['step']['to90']} 条")
        assert kept["success"] == 1.0, "筛选之后留出实体上仍然全对"

    print("\n  OK: 奖励只在结尾给时, 按锚状态分组把功劳落到具体动作上; 崩塌要靠输入依赖度查。")


if __name__ == "__main__":
    main()
