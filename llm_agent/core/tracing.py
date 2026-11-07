"""线程安全 span 树；默认不记录提示、工具参数或输出，避免追踪本身泄漏正文。

标准库教学格式，不声称实现 OTLP。时间用 monotonic，逻辑操作与 attempt 分开。
"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from time import monotonic
from uuid import uuid4
import json


class Tracer:
    def __init__(self, clock=monotonic):
        self.clock=clock
        self.spans=[]
        self._lock=RLock()
        self._current=ContextVar(f'span-{id(self)}',default=None)

    @contextmanager
    def span(self, kind, parent=None, operation='', attempt=1):
        with self._lock:
            parent=parent if parent is not None else self._current.get()
            if parent is not None and not any(s['id']==parent for s in self.spans):
                raise ValueError('父 span 不存在')
            trace=next((s['trace_id'] for s in self.spans if s['id']==parent),uuid4().hex)
            row={'id':len(self.spans)+1,'trace_id':trace,'parent':parent,'kind':kind,
                 'operation':operation,'attempt':attempt,'start':self.clock(),'end':None,'status':'running'}
            self.spans.append(row)
        token=self._current.set(row['id'])
        try:
            yield row
            if row['status']=='running':row['status']='ok'
        except BaseException as exc:
            row['status']='cancelled' if type(exc).__name__ in ('CancelledError','KeyboardInterrupt') else 'error'
            row['error_type']=type(exc).__name__  # 不收集可能含秘密的异常正文
            raise
        finally:
            row['end']=self.clock()
            self._current.reset(token)

    def export(self):
        with self._lock:
            return json.loads(json.dumps(self.spans))

    def summary(self):
        rows=self.export()
        if any(s['end'] is None for s in rows):raise ValueError('仍有未完成 span')
        roots=[s for s in rows if s['parent'] is None]
        return {'spans':len(rows),'roots':len(roots),
                'wall_seconds':sum(s['end']-s['start'] for s in roots),
                'summed_span_seconds':sum(s['end']-s['start'] for s in rows),
                'attempts':sum(s['kind']=='model_attempt' for s in rows),
                'logical_requests':len({(s['trace_id'],s['operation']) for s in rows if s['kind']=='model_attempt'}),
                'errors':sum(s['status']=='error' for s in rows)}
