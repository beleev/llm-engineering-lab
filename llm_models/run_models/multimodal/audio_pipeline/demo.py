"""真实合成波形→分帧→mel 特征，以及 PCM 量化还原与对齐。"""
import json
import numpy as np
MODES=("clean","noisy","coarse-codec")


def mel_features(wave,sr=8000,nfft=256,hop=80,bands=24):
    frames=np.lib.stride_tricks.sliding_window_view(wave,nfft)[::hop]*np.hanning(nfft)
    power=abs(np.fft.rfft(frames))**2
    mel=lambda f:2595*np.log10(1+f/700)
    hz=lambda m:700*(10**(m/2595)-1)
    knots=hz(np.linspace(mel(0),mel(sr/2),bands+2))
    freqs=np.fft.rfftfreq(nfft,1/sr)
    filters=np.array([np.maximum(0,np.minimum((freqs-a)/(b-a),(c-freqs)/(c-b)))
                      for a,b,c in zip(knots,knots[1:],knots[2:])])
    return np.log(np.maximum(power@filters.T,1e-8)),np.arange(len(frames))*hop/sr


def experiment(mode="clean"):
    if mode not in MODES:
        raise ValueError(mode)
    sr=8000
    t=np.arange(sr//2)/sr
    wave=.6*np.sin(2*np.pi*440*t)+.15*np.sin(2*np.pi*880*t)
    source=wave.copy()
    if mode=="noisy":
        source+=np.random.default_rng(9).normal(0,.15,len(t))
    levels=7 if mode=="coarse-codec" else 127
    tokens=np.rint(np.clip(source,-1,1)*levels).astype(np.int16)
    decoded=tokens/levels
    features,timestamps=mel_features(decoded)
    noise=np.mean((decoded-wave)**2)
    snr=10*np.log10(np.mean(wave**2)/max(noise,1e-12))
    assert np.isfinite(features).all()
    return {"metrics":{"波形样本":len(wave),"mel帧数":len(features),"mel通道":features.shape[1],
                       "还原SNR(dB)":round(float(snr),3),"帧步长(ms)":10,
                       "末帧起点(ms)":round(float(timestamps[-1]*1000),2)},
            "steps":["先量化并实际还原波形，再计算窗函数、短时傅里叶功率和三角 mel 滤波。",
                     "时间戳保留每帧起点；流式实现还需跨 chunk 保留 nfft-hop 个样本，不能重复或丢掉边界。",
                     "说话人分离、口音适应、背景噪声、ASR WER 与语音自然度是不同问题；低波形误差不代表理解正确。",
                     "这里 PCM 是可解码的最小 codec 基线；神经 codec 学习离散声学表征，不能直接把文本 token 当音频 token。"],
            "note":"真实数值信号处理，但输入是合成音调，不是语音。没有声纹识别、ASR/TTS 或神经 codec 能力；音频输出必须和说话人授权一起管理。"}


def main():
    rows={m:experiment(m) for m in MODES}
    assert rows["clean"]["metrics"]["还原SNR(dB)"]>rows["coarse-codec"]["metrics"]["还原SNR(dB)"]
    assert rows["clean"]["metrics"]["还原SNR(dB)"]>rows["noisy"]["metrics"]["还原SNR(dB)"]
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
