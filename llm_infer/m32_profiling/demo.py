"""真实 CPU profiler；GPU 分支显式选做，网页只回放稳定的事件统计。"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function

MODES=('eager','fused')

def measure(mode='eager',device='cpu',repeats=7,trace=None):
    if mode not in MODES or repeats<2:raise ValueError('mode/repeats')
    if device not in ('cpu','cuda'):raise ValueError('device')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA 不可用；不把 CPU 结果冒充 GPU')
    torch.set_num_threads(1);torch.manual_seed(8)
    x=torch.randn(128,64,device=device);w=torch.randn(64,64,device=device);b=torch.randn(64,device=device)
    def work():return x@w+b if mode=='eager' else torch.addmm(b,x,w)
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    for _ in range(3):work()
    sync()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    times=[]
    for _ in range(repeats):
        sync();start=time.perf_counter();result=work();sync();times.append((time.perf_counter()-start)*1e3)
    activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device=='cuda' else [])
    with profile(activities=activities,record_shapes=True,profile_memory=True) as prof:
        with record_function('linear_step'):result=work();sync()
    if trace:prof.export_chrome_trace(str(trace))
    events={e.key:e for e in prof.key_averages()}
    assert torch.allclose(result,x@w+b,atol=1e-5,rtol=1e-5)
    return {'device':device,'torch':torch.__version__,'python':platform.python_version(),'threads':torch.get_num_threads(),
            'warmup':3,'repeats':repeats,'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times),
            'logical_flops':2*128*64*64+128*64,'aten_calls':sum(e.count for k,e in events.items() if k.startswith('aten::')),
            'mm_calls':sum(e.count for k,e in events.items() if k in ('aten::mm','aten::addmm')),
            'cpu_self_us':sum(e.self_cpu_time_total for e in events.values()),
            'allocation_bytes':sum(max(0,e.self_cpu_memory_usage) for e in events.values()),
            'gpu_peak_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}

def experiment(mode='eager'):
    measured=measure(mode)
    return {'metrics':{'矩阵乘调用':measured['mm_calls'],'ATen 调用':measured['aten_calls'],
                       '逻辑 FLOPs':measured['logical_flops'],'CPU 分配字节':measured['allocation_bytes']},
            'steps':['真正执行 torch.profiler，记录算子、形状和内存事件。',
                     '先 warmup 3 次，再独立计时 7 次；耗时与环境由命令行 JSON 输出，网页不固化机器相关速度。',
                     f'{mode} 输出与 x@w+b 在固定容差内一致；事件数不等于 GPU kernel 数。'],
            'note':'CPU 分配字节是正向分配事件之和，不是峰值驻留内存。profiler 会扰动耗时，计时与采样分开；CPU self 时间排除子事件但不能直接与异步 GPU 总时间相加。CUDA 选做会同步后计时并记录峰值显存；本次默认只测 CPU。'}

def main(device='cpu', trace=None):
    print(json.dumps({m:measure(m,device,trace=f'{trace}-{m}.json' if trace else None) for m in MODES},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--trace');args=p.parse_args()
    main(args.device,args.trace)
