"""多轮消息→token→监督位置，角色错配与半条消息是失败对照。"""
import json
from llm_finetune.data.chat_template import Turn, serialize, training_example, collate, EOS


def experiment(mode='correct'):
    turns = [Turn('system','简洁回答'),Turn('user','查订单'),Turn('assistant','查询', 'c1','lookup'),
             Turn('tool','已送达','c1','lookup'),Turn('assistant','订单已送达'),
             Turn('user','谢谢'),Turn('assistant','不客气')]
    if mode == 'wrong-role':
        turns[3] = Turn('user','已送达')
    elif mode not in ('correct','truncated'):
        raise ValueError(mode)
    try:
        sample = training_example(turns, max_length=40 if mode == 'truncated' else 4096)
        ids, _ = serialize(turns)
        prefix, _ = serialize(turns[:2], generation=True)
        assert ids[:len(prefix)] == prefix
        short = training_example([Turn('user','好'),Turn('assistant','好')])
        batch = collate([sample, short])
        assert all(y == -100 for y, valid in zip(batch['labels'][1], batch['attention_mask'][1]) if not valid)
        metrics = {'token 数': len(ids), '监督 token 数': sum(x != -100 for x in sample['labels']),
                   '被监督 EOS 数': sample['labels'].count(EOS), '训练/推理前缀一致': True}
        detail = '三段 assistant 内容受监督；角色头、用户、工具结果和 padding 不计 loss。'
        detail += '\ntokens = ' + str(ids) + '\nlabels（已左移）= ' + str(sample['labels'])
    except ValueError as e:
        metrics, detail = {'输入通过校验': False}, str(e)
    return {'metrics':metrics,'steps':[detail]}


def main():
    rows = {m: experiment(m) for m in ['correct','wrong-role','truncated']}
    assert rows['correct']['metrics']['被监督 EOS 数']==2
    assert not rows['wrong-role']['metrics']['输入通过校验']
    assert not rows['truncated']['metrics']['输入通过校验']
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
