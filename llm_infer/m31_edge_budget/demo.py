"""端侧推理的内存、带宽、功耗和热约束：显式参数的预算模型。"""
import json
MODES=("fp16","int4","thermal")


def budget(bits=4,bandwidth=25e9,power=6.,context=4096):
    if bits not in (4,8,16) or bandwidth<=0 or power<=0 or context<1:
        raise ValueError("非法预算参数")
    params=1_000_000_000
    weight=params*bits/8
    # 24层，8个KV头，head_dim64，KV采用16位，与权重位宽分开。
    kv=2*24*8*64*context*2
    overhead=256*2**20
    seconds=max((weight+kv)/bandwidth,2*params/1e12)
    return {"权重MiB":round(weight/2**20,2),"KVMiB":round(kv/2**20,2),
            "总MiB":round((weight+kv+overhead)/2**20,2),
            "单token毫秒":round(seconds*1000,3),"单token焦耳":round(seconds*power,5),
            "可装入2GiB":weight+kv+overhead<=2*2**30}


def experiment(mode="int4"):
    if mode not in MODES:
        raise ValueError(mode)
    result=budget(bits=16 if mode=="fp16" else 4,bandwidth=12.5e9 if mode=="thermal" else 25e9)
    return {"metrics":result,
            "steps":["假设 1B 参数、25 GB/s 有效带宽、1 TFLOP/s 有效算力、6 W 系统功耗，全部是输入假设。",
                     "decode 时间按 max(计算时间, 权重与KV读取时间) 估计，另计256MiB运行时；不含量化解码开销。",
                     "热约束情形把有效带宽减半，模型位宽不变也会变慢；实际设备的算力/带宽/功耗变化须实测。",
                     "INT4 并未量化 KV。上下文、KV头数、并发和峰值中间张量仍可能让模型装不下。",
                     "部署选型要同步报告校准语料、质量切片、延迟、每次任务能耗与离线/隐私条件；低位宽不保证质量或能效改善。"],
            "note":"预算模型，不是手机或芯片跑分，不提供特定设备续航承诺。真实能耗需扣除/报告空闲基线并说明测量边界。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert not rows["fp16"]["metrics"]["可装入2GiB"]
    assert rows["int4"]["metrics"]["可装入2GiB"]
    assert rows["thermal"]["metrics"]["单token毫秒"] > rows["int4"]["metrics"]["单token毫秒"]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
