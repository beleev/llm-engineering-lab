"""Messages SSE 累积器：完整 message_stop 前不暴露可执行工具。

事件契约：https://platform.claude.com/docs/en/build-with-claude/streaming
未知顶层通知可忽略；未知内容/增量无法可靠回放，因此显式拒绝。
"""
import json


class StreamError(ValueError):
    pass


def _reject_constant(value):
    raise StreamError('工具参数含非法 JSON 数值: ' + value)


class MessageStream:
    def __init__(self, on_text=None):
        self.blocks={}; self.open=set(); self.fragments={}
        self.started=False;self.done=False;self.stop_reason=None;self.usage={}
        self.on_text=on_text

    def feed(self, event):
        if hasattr(event,'model_dump'):event=event.model_dump(exclude_none=True)
        kind=event['type']
        if self.done:raise StreamError('message_stop 之后仍有事件')
        if kind=='error':raise StreamError('服务端流式错误')
        if kind=='message_start':
            if self.started:raise StreamError('重复 message_start')
            self.started=True;self._usage(event['message'].get('usage',{}));return
        if kind=='ping':return
        if kind not in ('content_block_start','content_block_delta','content_block_stop','message_delta','message_stop'):
            return  # 前向兼容无关通知，不把未知数据当成工具参数
        if not self.started:raise StreamError('缺少 message_start')
        if kind=='content_block_start':
            i=event['index'];block=dict(event['content_block'])
            if i in self.blocks:raise StreamError('重复 block index')
            if block['type'] not in ('text','tool_use','thinking','redacted_thinking'):
                raise StreamError('未支持的内容块')
            self.blocks[i]=block;self.open.add(i);self.fragments[i]=''
        elif kind=='content_block_delta':
            i=event['index'];d=event['delta']
            if i not in self.open:raise StreamError('增量属于未打开的 block')
            block=self.blocks[i];t=d['type']
            field={'text_delta':'text','thinking_delta':'thinking','signature_delta':'signature'}.get(t)
            if t=='input_json_delta' and block['type']=='tool_use':
                self.fragments[i]+=d['partial_json']
            elif field and ((field=='text' and block['type']=='text') or (field in ('thinking','signature') and block['type']=='thinking')):
                block[field]=block.get(field,'')+d[field]
                if field=='text' and self.on_text:self.on_text(d[field])
            else:raise StreamError('内容块与增量类型不匹配')
        elif kind=='content_block_stop':
            i=event['index']
            if i not in self.open:raise StreamError('重复或孤立的 block_stop')
            block=self.blocks[i]
            if block['type']=='tool_use':
                if self.fragments[i]:
                    try:block['input']=json.loads(self.fragments[i], parse_constant=_reject_constant)
                    except json.JSONDecodeError as e:raise StreamError('工具参数 JSON 未完成') from e
                if not isinstance(block.get('input'),dict):raise StreamError('工具参数必须是对象')
                if not block.get('id') or not block.get('name'):raise StreamError('工具缺少 id/name')
            self.open.remove(i)
        elif kind=='message_delta':
            self.stop_reason=event.get('delta',{}).get('stop_reason',self.stop_reason)
            self._usage(event.get('usage',{}))
        else:
            if self.open or not self.stop_reason:raise StreamError('message_stop 时内容未闭合')
            self.done=True

    def _usage(self, usage):
        for key in ('input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens'):
            if key in usage:
                value=usage[key]
                if type(value) is not int or value<0:raise StreamError('非法 usage')
                self.usage[key]=value  # API 给的是累积值，不逐块相加

    def finish(self):
        if not self.done:raise StreamError('流在 message_stop 前结束，不能执行工具')
        ids=[b['id'] for b in self.blocks.values() if b['type']=='tool_use']
        if len(ids)!=len(set(ids)):raise StreamError('重复 tool_use id')
        return [self.blocks[i] for i in sorted(self.blocks)]
