// 阶段 3 · llm_train: 章节路由 -> 实验台 (train-moe-seq 另有 widgets 里的 MoeRouteLab / RingAttnLab)
const labs = {
  'train-batch-ddp': ['TrainBatchEquivLab'],
  'train-model-parallel': ['PipelineLab', 'TrainTpSplitLab'],
  'train-memory': ['TrainZeroMemoryLab', 'TrainActCkptLab'],
  'train-precision-stability': ['TrainFloatLineLab', 'TrainBlockScaleLab'],
  'train-moe-seq': ['TrainZigzagLab'],
  'train-collectives-loop': ['TrainRingAllReduceLab'],
  'train-pipeline-schedules': ['TrainInterleavedLab'],
  'train-lr-schedule': ['TrainLrScheduleLab'],
  'train-muon': ['TrainMuonLab', 'TrainMuonVsAdamLab'],
  'train-ulysses': ['TrainUlyssesLab'],
  'train-data-packing': ['TrainLshLab', 'TrainDataMixLab', 'TrainPackingLab'],
  'train-scaling': ['TrainIsoFlopLab', 'TrainMupLab'],
  'train-eval': ['TrainPassKLab', 'TrainJudgeLab'],
}

for (const route of ["train-model-parallel"]) labs[route] = [...(labs[route] || []), 'KnowledgeCompletionLab']
export default labs
