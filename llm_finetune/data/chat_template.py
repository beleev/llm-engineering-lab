"""教学聊天模板：结构化角色由模板写入，正文按 UTF-8 字节编码，不能冒充控制 token。

不是任何厂商的 tokenizer；用于核对多轮监督、工具配对和训练/推理格式一致性。
"""
from dataclasses import dataclass
import json

PAD, BOS, END, EOS = 0, 1, 2, 3
ROLES = {'system': 4, 'user': 5, 'assistant': 6, 'tool': 7}
HEADER_END = 8


@dataclass(frozen=True)
class Turn:
    role: str
    text: str
    call_id: str = ''
    tool_name: str = ''


def serialize(turns, generation=False, max_length=4096):
    """返回 token 与逐 token 监督标志。工具调用简化为每轮一个，不截断半条消息。"""
    if not turns:
        raise ValueError('至少需要一条消息')
    ids, supervise, pending, previous = [BOS], [False], None, None
    for index, turn in enumerate(turns):
        if turn.role not in ROLES or not isinstance(turn.text, str):
            raise ValueError('非法角色或正文')
        if turn.role == 'system' and index != 0:
            raise ValueError('system 只能出现在开头')
        if pending and (turn.role != 'tool' or (turn.call_id, turn.tool_name) != pending):
            raise ValueError('工具结果必须与待完成调用配对，不能当作新 user 消息')
        if turn.role == 'tool':
            if not pending:
                raise ValueError('孤立工具结果')
            pending = None
        elif turn.role == 'assistant':
            if previous not in ('user', 'tool'):
                raise ValueError('assistant 必须接在 user 或工具结果后')
            if bool(turn.call_id) != bool(turn.tool_name):
                raise ValueError('调用 id 与工具名必须同时出现')
            pending = (turn.call_id, turn.tool_name) if turn.call_id else None
        elif turn.role == 'user' and previous not in (None, 'system', 'assistant'):
            raise ValueError('重复用户轮次')
        elif turn.role not in ('assistant', 'tool') and (turn.call_id or turn.tool_name):
            raise ValueError('非工具角色不能携带调用 id')
        header = [ROLES[turn.role], HEADER_END]
        body = turn.text
        if turn.call_id:
            body = json.dumps({'id': turn.call_id, 'name': turn.tool_name, 'text': turn.text}, ensure_ascii=False, sort_keys=True)
        content = [32 + b for b in body.encode('utf-8')]
        end = EOS if turn.role == 'assistant' and not pending else END
        ids += header + content + [end]
        supervise += [False] * len(header) + [turn.role == 'assistant'] * (len(content) + 1)
        previous = turn.role
    if pending:
        raise ValueError('工具调用被截断，缺少结果')
    if generation:
        if previous not in ('user', 'tool'):
            raise ValueError('生成前缀必须停在待回复位置')
        ids += [ROLES['assistant'], HEADER_END]
        supervise += [False, False]
    if len(ids) > max_length:
        raise ValueError('超长消息：拒绝硬截断，先选择完整轮次或明确缩短正文')
    return ids, supervise


def training_example(turns, max_length=4096):
    ids, mask = serialize(turns, max_length=max_length)
    labels = [token if active else -100 for token, active in zip(ids[1:], mask[1:])]
    if all(x == -100 for x in labels):
        raise ValueError('没有 assistant 监督 token')
    return {'input_ids': ids[:-1], 'labels': labels}


def collate(examples):
    """右 padding 用于训练；loss mask 与 attention mask 分别保留。"""
    if not examples:
        raise ValueError('空 batch')
    width = max(len(x['input_ids']) for x in examples)
    return {key: [x[key] + [fill] * (width-len(x[key])) for x in examples]
            for key, fill in [('input_ids', PAD), ('labels', -100)]} | {
        'attention_mask': [[1]*len(x['input_ids'])+[0]*(width-len(x['input_ids'])) for x in examples]}
