"""真实字典缓存与 HMAC 命名空间；数值是确定性教学 KV 指纹，不执行模型。"""
import hashlib
import hmac
import json

MODES = ('shared', 'isolated', 'reuse', 'legacy-version', 'model', 'adapter', 'spoof')


class Gateway:
    def __init__(self):
        # 仅为可复现测试的固定凭证，真实系统需可信身份认证和独立密钥管理。
        self.identities = {'credential-A': 'A', 'credential-B': 'B'}
        self.secret = b'teaching-fixture-secret-not-a-production-key'

    def scope(self, credential, claimed_tenant=None):
        tenant = self.identities.get(credential)
        if tenant is None or (claimed_tenant is not None and claimed_tenant != tenant):
            raise PermissionError('租户声明与可信身份不一致')
        return hmac.new(self.secret, tenant.encode(), hashlib.sha256).hexdigest()


def compute(tokens, model, adapter):
    return hashlib.sha256(json.dumps([tokens, model, adapter]).encode()).hexdigest()


class Cache:
    def __init__(self, isolated=True, versioned=True):
        self.isolated, self.versioned = isolated, versioned
        self.items = {}

    def get(self, scope, tokens, model, adapter):
        # model/adapter 标识应绑定不可变内容摘要，不能只用可覆盖的显示名。
        identity = [scope if self.isolated else '', model if self.versioned else '',
                    adapter if self.versioned else '', tokens]
        key = hashlib.sha256(json.dumps(identity).encode()).digest()
        hit = key in self.items
        if not hit:
            self.items[key] = compute(tokens, model, adapter)
        return self.items[key], hit


def trial(mode):
    if mode not in MODES:
        raise ValueError(mode)
    gateway = Gateway()
    cache = Cache(isolated=mode != 'shared', versioned=mode != 'legacy-version')
    tokens = [1, 2, 3, 4]
    cache.get(gateway.scope('credential-A'), tokens, 'model-digest-v1', 'adapter-digest-v1')
    tenant = 'A' if mode in ('reuse','model','adapter','legacy-version') else 'B'
    model = 'model-digest-v2' if mode == 'model' else 'model-digest-v1'
    adapter = 'adapter-digest-v2' if mode in ('adapter','legacy-version') else 'adapter-digest-v1'
    try:
        scope = gateway.scope('credential-'+tenant, 'A' if mode == 'spoof' else None)
    except PermissionError:
        return {'hit': False, 'correct': True, 'rejected': True, 'cross_scope_reuse': False, 'cache_entries': len(cache.items)}
    value, hit = cache.get(scope, tokens, model, adapter)
    return {'hit': hit, 'correct': value == compute(tokens, model, adapter), 'rejected': False,
            'cross_scope_reuse': hit and tenant == 'B', 'cache_entries': len(cache.items)}


def experiment(mode='isolated'):
    r = trial(mode)
    return {'metrics': {'缓存命中': r['hit'], '返回值匹配模型约定': r['correct'],
                        '身份越权被拒绝': r['rejected'], '跨隔离域复用': r['cross_scope_reuse'],
                        '缓存条目': r['cache_entries'], '合成命中代价': None if r['rejected'] else (1 if r['hit'] else 5)},
            'steps': ['先用可信身份 A 写入 v1 模型 / adapter 的相同 token。',
                      '第二次请求改变租户、模型版本、adapter 版本或客户端租户声明。',
                      '既看是否命中，也将返回值与当前模型约定重新计算的指纹比较。'],
            'note': '命中 1 / 未命中 5 是合成代价，不是测量时延或攻击成功率。salt 只隔离缓存键，不隔离服务进程、网络和显存；多模态还要绑定内容摘要。'}


def main():
    for mode in MODES:
        print(json.dumps({'mode': mode, **experiment(mode)}, ensure_ascii=False))
    assert trial('shared')['cross_scope_reuse']
    assert not trial('isolated')['hit'] and trial('reuse')['hit']
    assert not trial('legacy-version')['correct']
    assert trial('spoof')['rejected']
    assert not trial('model')['hit'] and not trial('adapter')['hit']


if __name__ == '__main__':
    main()
