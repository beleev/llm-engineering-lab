"""磁盘检查点跨版本恢复；外部 SQLite 提交后子进程强退，恢复时读取收据。"""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory

MODES = ('compatible', 'new-flow', 'bad-node', 'bad-schema', 'bad-tool', 'revoked', 'crash')


def checkpoint(root, state):
    path = root / 'state.json'
    temp = root / 'state.tmp'
    with temp.open('w') as stream:
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def migrate(state):
    if type(state.get('schema_version')) is not int or state['schema_version'] not in (1, 2):
        raise ValueError('unsupported schema')
    if state.get('node') not in ('await_approval', 'submit', 'done'):
        raise ValueError('unknown saved node')
    if type(state.get('amount')) is not int or state['amount'] <= 0:
        raise ValueError('amount must be positive integer')
    if state.get('tool_version') != 1 or state.get('operation') != 'invoice-1':
        raise ValueError('tool contract changed')
    if type(state.get('flow_version')) is not int or state['flow_version'] not in (1, 2):
        raise ValueError('unknown behavior version')
    return {**state, 'schema_version': 2, 'summary': state.get('summary', '')}


def ledger(root, amount=None):
    with sqlite3.connect(root / 'external.db') as db:
        db.execute('CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, amount INTEGER NOT NULL)')
        if amount is not None:
            existing = db.execute('SELECT amount FROM effects WHERE key=?', ('invoice-1',)).fetchone()
            if existing and existing[0] != amount:
                raise ValueError('operation key reused with different arguments')
            db.execute('INSERT OR IGNORE INTO effects VALUES (?,?)', ('invoice-1', amount))
        return db.execute('SELECT count(*), coalesce(sum(amount),0) FROM effects').fetchone()


def resume(root, *, authorized=True, crash=False):
    state = migrate(json.loads((root / 'state.json').read_text()))
    if not authorized:
        raise PermissionError('恢复时重新授权失败')
    # 新流程增加规则；旧流程的运行不被静默改写，但权限仍重新检查。
    checked = state['flow_version'] == 2
    if checked and state['amount'] > 20:
        raise ValueError('v2 policy limit')
    count, total = ledger(root)
    reconciled = count > 0
    if reconciled and total != state['amount']:
        raise ValueError('外部收据参数不匹配')
    if not reconciled:
        ledger(root, state['amount'])
    if crash:
        os._exit(17)  # 真正退出进程：外部已提交，本地仍停在旧检查点。
    state.update(node='done', receipt='invoice-1', policy_checked=checked)
    checkpoint(root, state)
    return {'reconciled': reconciled, 'policy_checked': checked, 'schema_version': state['schema_version']}


def trial(mode):
    if mode not in MODES:
        raise ValueError(mode)
    with TemporaryDirectory(prefix='checkpoint-migration-') as directory:
        root = Path(directory)
        state = dict(schema_version=1, flow_version=1, node='await_approval', operation='invoice-1', amount=10, tool_version=1)
        if mode == 'new-flow':
            state.update(schema_version=2, flow_version=2)
        if mode == 'bad-node':
            state['node'] = 'renamed_removed_node'
        if mode == 'bad-schema':
            del state['amount']
        if mode == 'bad-tool':
            state['tool_version'] = 2
        checkpoint(root, state)
        exit_code = None
        if mode == 'crash':
            child = subprocess.run([sys.executable, '-m', 'llm_agent.m57_checkpoint_migration.demo', '--crash', str(root)], check=False, capture_output=True, timeout=20)
            exit_code = child.returncode
            assert exit_code == 17, child.stderr.decode()
            assert json.loads((root/'state.json').read_text())['node'] == 'await_approval'
        try:
            result = {**resume(root, authorized=mode != 'revoked'), 'rejected': False, 'reason': ''}
        except (ValueError, PermissionError) as error:
            result = {'rejected': True, 'reason': str(error), 'reconciled': False, 'policy_checked': False, 'schema_version': state['schema_version']}
        count, total = ledger(root)
        return {**result, 'writes': count, 'total': total, 'child_exit': exit_code}


def experiment(mode='compatible'):
    r = trial(mode)
    return {'metrics': {'拒绝恢复': r['rejected'], '外部提交笔数': r['writes'], '外部合计': r['total'],
                        '查询已有收据': r['reconciled'], '执行新业务规则': r['policy_checked'],
                        '状态格式版本': r['schema_version'], '故障子进程退出码': r['child_exit']},
            'steps': ['将暂停于审批节点的任务保存到磁盘；新版执行器先检查状态、节点、工具约定与执行权限。',
                      '格式版本决定如何读取状态；flow_version 决定继续哪套业务规则。',
                      r['reason'] or '提交与恢复均使用稳定操作键；外部已成功时核对收据，不重复产生副作用。'],
            'note': '真实文件、SQLite 和进程强退；不恢复任意 Python 栈，不实现分布式租约或任意 schema 自动迁移。原子重命名未对目录 fsync，不承诺断电持久性。'}


def main():
    for mode in MODES:
        print(json.dumps({'mode': mode, **experiment(mode)}, ensure_ascii=False))
    assert trial('crash')['writes'] == 1 and trial('crash')['reconciled']
    for mode in ('bad-node', 'bad-schema', 'bad-tool', 'revoked'):
        assert trial(mode)['writes'] == 0 and trial(mode)['rejected']


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--crash':
        resume(Path(sys.argv[2]), crash=True)
    else:
        main()
