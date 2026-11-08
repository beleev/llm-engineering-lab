"""生成序列一致性、像素水印和签名来源链的不同验收边界。"""
import hashlib
import hmac
import json
import numpy as np
MODES=("consistent","identity-drift","tampered")


def signed_manifest(content,key):
    digest=hashlib.sha256(content).hexdigest()
    return digest,hmac.new(key,digest.encode(),"sha256").hexdigest()


def valid_manifest(content,manifest,key):
    return hmac.compare_digest(signed_manifest(content,key)[1],manifest[1])


def experiment(mode="consistent"):
    if mode not in MODES:
        raise ValueError(mode)
    frames=np.zeros((8,16,24))
    for i in range(8):
        color=.8 if mode!="identity-drift" or i<4 else .3
        frames[i,6:10,2+i:6+i]=color
    identity=frames.max(axis=(1,2))
    centers=[float(np.where(frame>0)[1].mean()) for frame in frames]
    pattern=np.random.default_rng(17).choice([-1.,1.],size=frames.shape)
    watermarked=frames+.02*pattern
    key=b"PUBLIC-TEACHING-KEY-not-for-production"
    manifest=signed_manifest(watermarked.tobytes(),key)
    output=watermarked.copy()
    if mode=="tampered":
        output[4,6:10,6:10]=0
    correlation=float(np.mean((output-frames)*pattern))
    return {"metrics":{"最大身份强度跳变":round(float(abs(np.diff(identity)).max()),4),
                       "每帧位移":np.diff(centers).tolist(),"水印相关":round(correlation,5),
                       "来源签名通过":valid_manifest(output.tobytes(),manifest,key)},
            "steps":["从实际帧计算运动和外观差异；一致运动不保证身份一致，也不证明物理世界模型正确。",
                     "加性扩频水印用相关检测；此例知道未加水印参考帧，真实盲检更难，必须测误报与变换攻击。",
                     "签名链绑定具体内容摘要；篡改使验证失败。合法编辑应创建新摘要并记录父版本，而不是假装原签名仍有效。",
                     "音画同步应对齐事件时间戳；镜头运动与物体运动需区分。深伪分类器不能替代来源与授权核验。"],
            "note":"合成短片与公开演示密钥，不是安全水印、C2PA 实现或通用深伪检测器；签名验证的是完整性与密钥持有者，不证明内容真实。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["consistent"]["metrics"]["最大身份强度跳变"]==0
    assert rows["identity-drift"]["metrics"]["最大身份强度跳变"]>0
    assert not rows["tampered"]["metrics"]["来源签名通过"]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
