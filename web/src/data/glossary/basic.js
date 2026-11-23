// 阶段 1 · llm_basic 术语。number 取自代码里的公式或实验台里真算出来的数。
const t = (term, aka, oneliner, number, route) => ({ term, aka, stage: 'basic', oneliner, number, route })

export default [
  t('字符级 tokenizer', 'char-level', "将语料中出现过的每个唯一字符直接映射为一个整数 id 的最简分词器; 本章主训练流水线即采用该方案。", "词表大小 vocab 为 65", 'basic-data'),
  t('BPE', 'Byte Pair Encoding', "从 256 个基础字节出发迭代合并最高频相邻对的子词分词算法; 本仓库中作为独立 demo 实现。", "60 字符压至 24 token", 'basic-data'),
  t('next-token prediction', '下一词预测', "标签序列 y 为输入序列 x 右移一位的自回归训练目标; 单条长 T 的序列可并行提供 T 个预测样本。", "y = data[i+1 : i+T+1]", 'basic-data'),
  t('cache', '手写 autograd tape', "手写反向传播的前向缓存约定: forward 仅保存反向求导必需的中间张量, 并将其打包为 tuple 随输出一并返回。", "forward 返回 (out, cache)", 'basic-forward'),
  t('Pre-LN 残差', 'pre-norm residual', "形如 $x + f(\\mathrm{norm}(x))$ 的残差连接: 归一化置于子层分支入口, 残差主干保持纯加法以使梯度无损直达底层。", "$d(x+f)/dx = 1 + f^{\\prime}$", 'basic-forward'),
  t('causal mask', '因果掩码', "屏蔽未来位置的注意力掩码: 限制位置 $i$ 仅读取 $\\le i$ 的历史位置, 使长 $T$ 的序列在训练时并行计算。", "上三角填入 −inf", 'basic-forward'),
  t('[B,T,T]', 'attention 分数矩阵', "标准 Transformer 中唯一跨序列位置交互信息的注意力权重张量, 其元素规模随序列长度呈 $T^2$ 二次增长。", "$T=64$ 占激活 14%, $T=256$ 占 40%", 'basic-forward'),
  t('激活显存', 'activation memory', "前向传播为支持反向求导而驻留的中间激活张量总开销; 在本章默认配置下约为模型参数体积的 60 倍。", "激活 21.0 MiB 对比参数 356 KiB", 'basic-forward'),
  t('n_layer 循环', '堆层', "因每个 Transformer block 输入与输出张量同形, 堆叠深度仅需单层 for 循环, 层数可直接由参数字典键名推断。", "默认 1 层, 可由 --n-layer 调节", 'basic-forward'),
  t('链式法则', 'chain rule', "反向传播的求导法则: 当前节点梯度等于上游传入梯度与局部导数的乘积, 从标量损失对自身导数 1 出发逐层回传。", "$\\partial L/\\partial x = W^\\top \\cdot \\partial L/\\partial z$", 'basic-backward'),
  t('softmax+CE 梯度', 'p − onehot', "将 softmax 与交叉熵合并求导后分母项 $1/p[y]$ 恰好相互抵消, 既化简反向公式又避免极小概率引发数值溢出。", "$\\mathrm{dlogits} = (p - \\mathrm{onehot})/N$", 'basic-backward'),
  t('softmax 雅可比', 'Jacobian', "因归一化分母使每个输出依赖全部输入, 局部导数为稠密矩阵而非对角阵, 非对角部分即跨坐标「耦合项」。", "$J = \\mathrm{diag}(p) - pp^\\top$", 'basic-backward'),
  t('np.add.at', '重复索引累加', "同一 token 在单次 batch 内重复出现时, 回传至对应 embedding 行的梯度必须逐次无缓冲累加而非覆盖。", "直接写 dW[ids] += 会丢失重复索引梯度", 'basic-backward'),
  t('gradcheck', '数值梯度检查', "利用有限中心差分近似导数以校验手写反向传播正确性的方法: 先逐算子遍历全部元素核验, 再对整网参数抽样复核。", "逐算子相对误差 rel err 为 1e-11~2e-10", 'basic-backward'),
  t('中心差分', 'central difference', "在待测点两侧各偏移 $\\varepsilon$ 求差商; 总误差由 $\\varepsilon^2$ 截断项与 $\\delta/\\varepsilon$ 舍入项叠加而呈 U 形曲线。", "float64 下误差谷底位于 $\\varepsilon \\approx 10^{-5}$", 'basic-backward'),
  t('RMSNorm 耦合项', 'coupling term', "RMSNorm 约束每行均方恒定, 增大单个 $x_i$ 会压缩其余输出; 反向求导漏掉该耦合项会导致 gradcheck 报错。", "漏掉后 rel err 升至 3.11e-01", 'basic-backward'),
  t('Adam', '自适应矩估计', "同时维护梯度指数移动一阶矩与二阶矩的优化器: 用一阶矩平滑更新方向, 用二阶矩平方根逐坐标缩放自适应步长。", "每步有效参数位移量级约为 lr", 'basic-optim-sample'),
  t('偏差修正', 'bias correction', "因矩估计 $m$ 与 $v$ 从零初始化而在训练初期偏向于零, 需分别除以 $1 - \\beta^t$ 以恢复无偏估计量级。", "$\\hat m = m / (1 - \\beta_1^t)$", 'basic-optim-sample'),
  t('AdamW', '解耦权重衰减', "将权重衰减项 $\\lambda W$ 直接施加于参数更新步而非混入梯度, 使其不受二阶矩因子 $1/\\sqrt{v}$ 缩放。", "1 维 gain 与 bias 参数不施加衰减", 'basic-optim-sample'),
  t('全局范数裁剪', 'grad clip', "按全体参数梯度的联合全局范数统一乘以同一缩放系数限制幅值, 裁剪后整体梯度方向保持不变。", "$g \\leftarrow g \\cdot \\min(1, \\mathrm{max\\_norm}/\\|g\\|)$", 'basic-optim-sample'),
  t('条件数', 'condition number κ', "损失曲面最陡方向与最平坦方向的曲率之比; 普通 SGD 的最大稳定学习率受最陡方向限制而在平坦方向收敛缓慢。", "SGD 稳定收敛要求 $\\mathrm{lr} < 2/\\kappa$", 'basic-optim-sample'),
  t('temperature', '采样温度', "计算 $\\mathrm{softmax}(z/T)$ 前对 logits 施加的温度缩放系数: $T<1$ 使概率分布更尖锐确定, $T>1$ 使分布更平缓随机。", "$T \\to 0$ 退化为贪心解码", 'basic-optim-sample'),
  t('top-k 采样', 'top-k sampling', "每步仅保留预测概率最高的 k 个候选 token 重新归一化后采样, 截断低概率长尾以减少异常离题输出。", "$k = 1$ 等价于贪心解码", 'basic-optim-sample'),
{
  "term": "感受野",
  "aka": "Receptive field",
  "stage": "basic",
  "oneliner": "网络结构上单个输出位置所能读取的输入位置集合; 结构可达仅代表信息通路存在, 并不保证训练后能有效利用。",
  "number": "纯滑窗: min(t+1, 1+L×(W−1)), W 含当前位置",
  "route": "basic-forward"
},
]
