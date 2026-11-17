"""web/src/utils/zhPunct.js 的 Python 版, 规则逐条照抄。浏览器回归脚本用它把期望文字转成页面上显示的样子
(数据里写半角, 页面显示中文标点), 再按文字找元素。两边规则改一处要同步改另一处。"""
import re

HAN = re.compile(r'[㐀-鿿]')
ID_TAIL = re.compile(r'[A-Za-z0-9_\]]')
AFTER_Q = re.compile(r'[㐀-鿿」）』"\'”’]')
MAP = {',': '，', ';': '；', ':': '：'}


def zh(s):
    if not isinstance(s, str) or not HAN.search(s):
        return s
    out, stack, i = '', [], 0
    while i < len(s):
        c = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else None
        in_code = 'code' in stack
        if c == '(':
            code = in_code or (i > 0 and ID_TAIL.match(s[i - 1]) is not None)
            stack.append('code' if code else 'prose')
            if code:
                out += c
            else:
                if out.endswith(' '):  # 「词 (」→「词（」
                    out = out[:-1]
                out += '（'
        elif c == ')':
            top = stack.pop() if stack else None
            if top == 'code':
                out += c
            else:
                out += '）'
                if nxt == ' ':
                    i += 1
        elif c in '[{':
            stack.append('code')
            out += c
        elif c in ']}':
            if stack and stack[-1] == 'code':
                stack.pop()
            out += c
        elif not in_code and c in MAP and nxt in (' ', None):
            out += MAP[c]
            if nxt == ' ':
                i += 1
        elif not in_code and c in '?!' and (nxt is None or nxt == ' ' or AFTER_Q.match(nxt)):
            out += '？' if c == '?' else '！'
            if nxt == ' ':
                i += 1
        else:
            out += c
        i += 1
    return out


if __name__ == '__main__':
    assert zh('分页 (PagedAttention) 借的是, 结论: 省显存; 对吗?') == '分页（PagedAttention）借的是，结论：省显存；对吗？'
    assert zh('调用 next(messages, tool_schemas) 后') == '调用 next(messages, tool_schemas) 后'
    assert zh('帧步长(ms)') == '帧步长（ms）'
    assert zh('pure english, ok') == 'pure english, ok'
    print('zh_punct: 通过')
