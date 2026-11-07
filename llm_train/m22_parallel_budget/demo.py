"""DP×TP×PP 的 rank/内存/时间预算；显式代价模型，不启动 GPU 或测量 MFU。"""
import json
from math import ceil


def estimate(dp=2,tp=2,pp=2,*,world=8,batch=64,layers=24,hidden=2048,
             params=1_000_000_000,seq=1024,micro=4,overlap=True,latency_us=8):
    if any(type(v)!=int or v<=0 for v in (dp,tp,pp,world,batch,layers,hidden,params,seq,micro)):
        raise ValueError('维度必须是正整数')
    if dp*tp*pp!=world or hidden%tp or layers%pp or batch%(dp*micro):
        raise ValueError('要求 DP×TP×PP=world，hidden/TP、layers/PP、batch/(DP×micro) 可整除')
    grid=[{'rank':(d*pp+p)*tp+t,'dp':d,'tp':t,'pp':p}
          for d in range(dp) for p in range(pp) for t in range(tp)]
    microbatches=batch//(dp*micro)
    # BF16 参数+梯度，FP32 主参数+Adam m/v，共16 B/参数；不含 ZeRO。
    state=params*16/(tp*pp)
    activation=2*seq*hidden*micro*(layers/pp)*min(pp,microbatches)/tp
    flops=6*params*batch*seq/world
    peak=100e12; hbm=1.5e12; link=50e9; alpha=latency_us/1e6
    # 内存流量粗估为每 microbatch 扫三遍本地参数状态+激活；单位 bytes。
    traffic=3*state*microbatches+activation*microbatches
    compute=max(flops/peak,traffic/hbm)
    intensity=flops/traffic
    dp_messages=2*(dp-1)
    dp_time=dp_messages*alpha+2*(dp-1)/dp*(params*2/(tp*pp))/link
    tp_messages=4*(layers//pp)*microbatches*2*(tp-1)
    tp_time=tp_messages*alpha+4*(layers//pp)*microbatches*2*(tp-1)/tp*(micro*seq*hidden*2)/link
    pp_messages=2*microbatches if pp>1 else 0
    pp_time=pp_messages*(alpha+micro*seq*hidden*2/(tp*link))
    # TP 与 PP 依赖当前层结果，不能随意藏进计算；这里只允许 DP 梯度通信重叠。
    stage=compute+tp_time+pp_time
    steady=max(stage,dp_time) if overlap else stage+dp_time
    bubble=(pp-1)/microbatches
    seconds=steady*(1+bubble)
    return {'grid':grid,'state_gib':state/2**30,'activation_gib':activation/2**30,
            'microbatches':microbatches,'compute_ms':compute*1000,'dp_ms':dp_time*1000,
            'tp_ms':tp_time*1000,'pp_ms':pp_time*1000,'bubble_ratio':bubble,
            'step_ms':seconds*1000,'tokens_per_second':batch*seq/seconds,
            'modeled_mfu':flops/(seconds*peak),'arithmetic_intensity':intensity,
            'roofline_flops_per_second':min(peak,intensity*hbm),
            'messages':dp_messages+tp_messages+pp_messages}


def experiment(mode='balanced'):
    configs={'balanced':dict(dp=2,tp=2,pp=2),'more-tp':dict(dp=1,tp=4,pp=2),
             'no-overlap':dict(dp=2,tp=2,pp=2,overlap=False),
             'high-latency':dict(dp=1,tp=4,pp=2,latency_us=100),
             'invalid':dict(dp=2,tp=3,pp=2)}
    if mode not in configs:raise ValueError(mode)
    try:r=estimate(**configs[mode])
    except ValueError as exc:
        return {'metrics':{'配置合法':False},'steps':[str(exc)],'note':'约束检查先于时间估算。'}
    return {'metrics':{'状态 GiB/卡':round(r['state_gib'],3),'激活 GiB/卡':round(r['activation_gib'],3),
                       '步时间 ms（估算）':round(r['step_ms'],2),'tokens/s（估算）':round(r['tokens_per_second']),
                       'MFU（估算）':round(r['modeled_mfu'],3),'通信次数/关键卡':r['messages']},
            'steps':[f"rank {g['rank']}: DP={g['dp']} / TP={g['tp']} / PP={g['pp']}" for g in r['grid']]+
                    [f"计算/访存下界 {r['compute_ms']:.2f} ms；DP {r['dp_ms']:.2f}；TP {r['tp_ms']:.2f}；PP {r['pp_ms']:.2f}",
                     f"流水线气泡倍率 1+{r['bubble_ratio']:.3f}；算术强度 {r['arithmetic_intensity']:.1f} FLOP/byte"],
            'note':'固定 1B 参数、batch=64、8 卡；100 TFLOP/s、HBM 1.5 TB/s、链路 50 GB/s。无实测；忽略拓扑争用、临时缓冲和重算。'}

def main():
    a=estimate(); b=estimate(dp=1,tp=4,pp=2)
    assert len({g['rank'] for g in a['grid']})==8
    assert b['state_gib']<a['state_gib'] and b['step_ms']>a['step_ms']
    assert estimate(overlap=False)['step_ms']>=a['step_ms']
    assert experiment('invalid')['metrics']['配置合法'] is False
    print(json.dumps({m:experiment(m) for m in ('balanced','more-tp','high-latency')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
