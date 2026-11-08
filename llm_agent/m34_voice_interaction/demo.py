"""语音事件状态机: VAD / ASR / LLM / TTS 的时间边界。文本转录是夹具, 不冒充 ASR。"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math


@dataclass(frozen=True)
class Event:
    at: int
    kind: str
    generation: int = 0
    text: str = ""


EVENTS = (Event(0, "speech-start"), Event(100, "asr-partial", 1, "查"),
          Event(180, "asr-final", 1, "查订单"), Event(190, "fast-ack", 1, "正在查"),
          Event(220, "tts-echo", 1), Event(250, "speech-start"),
          Event(270, "tts-chunk", 1, "旧结果"), Event(300, "asr-final", 2, "停止查询"),
          Event(320, "slow-result", 1, "订单已发货"), Event(340, "tts-chunk", 2, "已停止"))


def rms(samples):
    return math.sqrt(sum(x*x for x in samples) / max(1, len(samples)))


def voice_activity(samples, playback_reference):
    if len(samples) != len(playback_reference):
        raise ValueError("回声参考必须逐样本对齐")
    # 已知回声参考的减法只解释机制, 不替代真实声学回声消除。
    return rms([a-b for a, b in zip(samples, playback_reference)]) > .05


def experiment(mode="duplex"):
    if mode not in {"half", "duplex", "no-generation"}:
        raise ValueError(mode)
    generation, speaking, output, stale, trace = 0, False, [], 0, []
    for event in EVENTS:
        accepted = True
        if event.kind == "speech-start":
            if speaking and mode == "half":
                accepted = False
            else:
                generation += 1
                speaking = False
        elif event.kind == "tts-echo":
            accepted = False
        elif event.kind in {"fast-ack", "tts-chunk", "slow-result"}:
            accepted = mode == "no-generation" or event.generation == generation
            if accepted:
                output.append(event.text)
                stale += event.generation < generation
                speaking = True
        elif event.generation != generation:
            accepted = False
        trace.append({"title": f"{event.at} ms · {event.kind}", "detail": f"generation={generation}; {'接收' if accepted else '丢弃'} {event.text}",
                      "ok": not (accepted and event.generation and event.generation < generation)})
    return {"output": output, "stale": stale, "generations": generation,
            "interrupted": generation == 2, "first_ack_ms": 190, "trace": trace}


def main():
    rows = {m: experiment(m) for m in ("half", "duplex", "no-generation")}
    assert rows["duplex"]["output"] == ["正在查", "已停止"]
    assert rows["duplex"]["stale"] == 0 and rows["duplex"]["interrupted"]
    assert rows["no-generation"]["stale"] == 2 and not rows["half"]["interrupted"]
    echo = [.1, -.1] * 40
    assert not voice_activity(echo, echo)
    assert voice_activity([x + .2 for x in echo], echo)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
