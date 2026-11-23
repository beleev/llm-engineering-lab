"""
微调方法: 每个文件一种方法, 可以独立阅读。

    监督 / 参数高效:  sft · lora · dora · qlora
    模型合并:        merge (Task Arithmetic / TIES / DARE / SLERP, 只动权重, 不训练)
    离线偏好:        dpo · kto (单条好 / 坏标签) · simpo · orpo · reward_model
                     rlaif (偏好对由规则 judge 按 "宪法" 自动造)
    过程打分:        prm (每一步一个分的 PRM, 对照只看最终答案的 ORM)
    在线 RL:         ppo (带 critic) · grpo (含 DAPO / Dr.GRPO / GSPO 开关)
                     is_correction (训练-推理不一致的 TIS / MIS 校正)
    量化训练:        qat (假量化 + STE, 分组对称低比特)
    蒸馏:            distill (off-policy, forward KL) · on_policy_distill (on-policy, reverse KL)

共 18 个文件。每个文件头固定有一行 "与论文的差异" 或 "未实现", 写明本库简化了什么。
"""
