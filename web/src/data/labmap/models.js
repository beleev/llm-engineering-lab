// 阶段 2 · llm_models: 章节 -> 实验台 (掩码在 Attention，MTP 在 models-mtp)
export default {
  'models-generation': ['ModelKvGenLab'],
  'attention': ['ModelQkNormLab'],
  'position': ['ModelRopeScaleLab'],
  'models-mamba': ['ModelMambaLab', 'ModelLinearAttnLab'],
  'models-moe-balance': ['ModelAuxFreeLab'],
  'models-dsa': ['ModelDsaLab'],
  'models-gptoss': ['ModelSwaLayersLab', 'ModelAttnSinkLab'],
  'models-llada': ['ModelLladaLab'],
  'models-var': ['ModelVarLab'],
  blocks: ['ModelResidualLab', 'ModelInitLab'], // 绑权重 + 初始化尺度: 为什么首步 loss 应该等于 ln V
}
