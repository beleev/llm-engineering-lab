"""
m03 demo — Continuous Batching 调度器 (mock 模型: 只看调度, 不看 token 内容)

运行: python -m llm_infer.m03_continuous_batching.demo
看什么: [1] 宽松 pool 下每步 batch 的组成 (P=prefill n 个 token, D=decode) 和 pool 占用;
        [2] 紧张 pool 下的抢占, 以及两条必须成立的性质:
    不活锁 —— 队首请求拿不到 block 时必须落到 decode;  抢占不丢不多 —— 不多生成也不丢 token。
"""
from __future__ import annotations

from llm_infer.core.utils import banner, kv
from llm_infer.m03_continuous_batching.scheduler import Scheduler, SchedulerConfig

REQUESTS = [([1, 2, 3, 4, 5], 8), ([10, 20, 30], 4), ([7, 8, 9, 10, 11, 12, 13], 10),
            ([100, 101], 6), ([50, 51, 52, 53, 54], 5)]          # (prompt, max_new)


def run(cfg: SchedulerConfig, verbose: bool = True, max_steps: int = 500):
    """把 REQUESTS 全部提交, 循环 schedule → mock 模型 → postprocess 直到跑完。

    返回 (调度器, 各请求的 Sequence, 总步数)。max_steps: 超过就判定为活锁。
    """
    sched = Scheduler(cfg)
    seqs = [sched.add_request(p, m, eos_id=-1) for p, m in REQUESTS]
    step = 0
    while sched.has_unfinished():
        step += 1
        assert step <= max_steps, f"活锁: 跑了 {max_steps} 步仍有未完成请求, 调度器在空转"
        batch = sched.schedule()
        assert batch, "有未完成请求却调度出空 batch (队首进不来时应落到 decode)"
        # mock 模型: 第 k 个输出 token 的值就是 k → 抢占重算后内容是否连续一眼可查
        # 只有 KV 追平 (num_computed + n == num_tokens) 的序列才有新 token, prefill 中途的给 None
        toks = [seq.num_output if seq.num_computed + n == seq.num_tokens else None for seq, n in batch]
        desc = " ".join(f"s{seq.seq_id}:{'P' + str(n) if n > 1 else 'D'}" for seq, n in batch)
        done = sched.postprocess(batch, toks)
        if verbose:
            print(f"  step {step:>2}  [{desc:<28}] pool={sched.bm.stats()['utilization']:>6} "
                  f"preempt={sched.preempt_count}" + (f"  ✓完成 {[s.seq_id for s in done]}" if done else ""))
    return sched, seqs, step


def main():
    banner("M03 - Continuous Batching")
    for p, m in REQUESTS:
        print(f"  request: prompt_len={len(p)} max_new={m}")

    loose = SchedulerConfig(max_batch_seqs=4, max_batch_tokens=64, block_size=4, num_blocks=32)
    print(f"\n[1] 宽松 pool ({loose.num_blocks} blocks × {loose.block_size}): 请求随到随进, 各自完成各自退出")
    sched, seqs, steps_loose = run(loose)
    assert sched.preempt_count == 0, f"宽松 pool 不应发生抢占, 实际 {sched.preempt_count} 次"

    tight = SchedulerConfig(max_batch_seqs=4, max_batch_tokens=64, block_size=4, num_blocks=7)
    print(f"\n[2] 紧张 pool ({tight.num_blocks} blocks × {tight.block_size} = {tight.num_blocks * tight.block_size} token): "
          "触发抢占; 队首进不来时 running 照常 decode, 不会活锁")
    sched, seqs, steps_tight = run(tight)

    banner("结果")
    kv("总 step (宽松 / 紧张)", f"{steps_loose} / {steps_tight}")
    kv("抢占次数", sched.preempt_count)
    kv("各序列被抢占次数", [s.num_preempted for s in seqs])
    assert sched.preempt_count > 0, "紧张 pool 应触发抢占, 否则这一节什么也没测到"
    for seq, (_, max_new) in zip(seqs, REQUESTS):
        # 抢占后: 输出一个不多 (max_new 不被重置)、一个不少、顺序不乱
        assert seq.output_ids == list(range(max_new)), \
            f"seq {seq.seq_id} 的输出应是 0..{max_new - 1} 各一个, 实际 {seq.output_ids}"
    assert sched.bm.num_free_blocks() == tight.num_blocks, \
        f"block 泄漏: 全部跑完后 {tight.num_blocks} 个 block 应都空闲, 实际 {sched.bm.num_free_blocks()}"
    assert sum(sched.bm.ref_count) == 0, f"block 泄漏: 引用计数应全为 0, 实际 {sched.bm.ref_count}"
    print("  ✓ 无活锁; 抢占前后输出完整且恰好 max_new 个; block 全部归还")


if __name__ == "__main__":
    main()
