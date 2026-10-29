# Scaling Laws：从预测 Loss 到指导预训练

> 核心问题：给定有限算力、数据和模型设计，如何选择模型规模、训练 token、重复次数与优化超参数？Scaling law 是特定实验条件下拟合的经验规律，不是对所有模型都成立的自然定律。

## 1. 研究主线

1. **描述**：Loss 如何随模型、数据与算力变化？
2. **分配**：固定 FLOPs，如何选择参数量 \(N\) 与训练 token 数 \(D\)？
3. **修正现实约束**：数据有限、重复曝光、Batch Size、学习率和训练时长如何改变最优配置？
4. **迁移到真实系统**：Dense/MoE、稀疏度、数据来源与训练 recipe 是否改变 scaling 规律？

区分三个目标：Loss scaling 预测性能；compute-optimal scaling 决定固定算力下的 \(N,D\) 分配；hyperparameter scaling 预测学习率、Batch Size 等配置。三者相关，但不可互相替代。

## 2. Kaplan (2020)：建立经验幂律

论文：[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

在不同参数量 \(N\)、数据规模 \(D\) 与训练算力 \(C\) 下训练语言模型，拟合验证集 Loss。便于理解的简化形式是：

\[
L(N,D)\approx L_\infty+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
\]

\(L_\infty\) 是不可约 Loss，另外两项描述模型容量与数据不足带来的 excess loss。上式是示意形式，并非论文所有拟合式的完整复现。

**为什么是幂律？** 实验上，跨越很大的规模范围时，Loss 与规模的对数关系近似线性，即 \(L-L_\infty\propto X^{-\alpha}\)，因此小模型拟合可用于预测大模型。这是经验规律，不是由 Central Limit Theorem 直接推出的定理；CLT 的 \(O(n^{-1/2})\) 误差收敛不能单独证明神经网络 Loss 必须服从幂律。

Kaplan 的关键结论是：按其拟合与训练设定，固定算力下倾向于训练更大的模型、使用相对较少的数据，并在完全收敛前停止。

## 3. Chinchilla (2022)：固定算力下重新分配模型与数据

论文：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

Hoffmann et al. 通过近似等算力实验比较不同模型规模与 token 数组合的最终 Loss，再拟合 compute-optimal 配置。常用 Dense Transformer 算力近似为：

\[
C\approx 6ND,
\]

其中 \(N\) 为参数量，\(D\) 为训练 token 数；这是粗略估计，具体架构会带来 FLOPs 偏差。

若 \(N_{\mathrm{opt}}\propto C^a\)、\(D_{\mathrm{opt}}\propto C^b\)，则由 \(ND\propto C\) 得 \(a+b\approx1\)。Chinchilla 的经验结果接近 \(a\approx b\approx1/2\)：算力增长时，模型和 token 数都应增加，而非只扩大模型。

| 维度 | Kaplan | Chinchilla |
|:--|:--|:--|
| 核心问题 | Loss 如何随规模变化 | 固定算力下怎样选择模型与数据 |
| 主要建议 | 更大模型、相对少数据、较早停止 | 模型与 token 数都应充分扩展 |
| 方法 | 拟合规模关系与训练动态 | IsoFLOP 实验寻找最低 Loss |
| 边界 | 依赖拟合与实验设定 | 依赖数据、训练 recipe 与预算范围 |

**Chinchilla 的 token/参数比例不是普适常数。** 它是在特定模型族、数据与训练预算下拟合出的经验结论。

### IsoFLOP 怎么做？

固定算力预算 \(C\)，由于 \(C\approx6ND\)，增大模型就必须相应减少 token 数。对每个算力水平，训练不同 \(N,D\) 组合、比较最终 Loss，找到该预算下的最低点，再拟合不同预算下的最优点。它回答的是**同一份算力应该如何花**，而不是更大的模型是否总是更好。

## 4. Batch Size：Critical Batch Size 与 \(S_{\min}\)

Batch Size 并非越大越好。增加 batch 可减少梯度噪声、提高并行度，但超过一定尺度后，额外 token 带来的优化收益会下降。

- **Critical Batch Size \(B_{\mathrm{crit}}\)**：优化效率开始出现收益递减的特征尺度，通常由不同 batch 下的训练进展或梯度噪声/效率曲线估计，不是固定常数。
- **\(S_{\min}\)**：理想化无限大 batch 假设下，达到相同优化进展所需的最少更新步数；具体定义依论文模型而定。
- **Batch-size efficiency correction**：比较真实 batch 与理想化极限下的训练进展，估计增加的 token 并非都等价地转化为优化收益。

Chinchilla 重点是最终 Loss 的 compute-optimal \(N,D\) 分配，不应简单说成沿用了 Kaplan 的全部 batch 校正公式。实验中必须同时记录 batch、训练步数和 token 数。

参考：[Kaplan](https://arxiv.org/abs/2001.08361)，[Stanford CS336](https://stanford-cs336.github.io/) Lecture 09 / Lecture 11。

## 5. 数据有限：重复数据与 Data-Constrained Scaling

- [Scaling Laws and Interpretability of Learning from Repeated Data (2022)](https://arxiv.org/abs/2205.10487)：重复曝光仍能推动优化，但边际收益会下降，也可能加重过拟合。必须区分 unique tokens（独立 token）和 total training tokens（总曝光 token）。
- [Scaling Data-Constrained Language Models (2023)](https://arxiv.org/abs/2305.16264)：数据受限、必须重复训练时，最优配置不能只由 \(N,D,C\) 决定，还要考虑数据集大小与重复程度。
- [Prescriptive Scaling Laws for Data Constrained Training (2026)](https://arxiv.org/abs/2605.01640)：在 Chinchilla 式框架中显式加入重复曝光的过拟合惩罚，以拟合系数描述重复数据代价。作者报告强权重衰减能显著降低该过拟合系数（案例中约降低 70%）。

**发展关系：** Kaplan/Chinchilla 刻画模型、数据与算力；后续工作进一步纳入数据独立性和重复曝光。高质量数据不足时，直接外推经典 token/参数比例可能误导配置选择。

## 6. 真实 recipe 与训练时长

- [DeepSeek LLM (2024)](https://arxiv.org/abs/2401.02954)：把数据构建、训练设置、模型规模和评测放进真实开源预训练配方中。应区分观测到的 recipe 与可普遍外推的规律。
- [Chinchilla Scaling: A Replication Attempt (2024)](https://arxiv.org/abs/2404.10102)：检验 compute-optimal 结论在不同条件下能否复现。偏差可能来自算力范围、数据质量、tokenizer、优化器、训练步数或实验噪声。
- [Scaling Optimal LR Across Token Horizons (2024)](https://arxiv.org/abs/2409.19913)：最优学习率会随 token horizon（计划训练 token 总量/时长）变化。短 horizon 找到的 LR 不一定适合长训练；LR schedule、总 token 数与目标 checkpoint 应联合考虑。

## 7. Hyperparameter Scaling：Step Law

论文：[Predictable Scale: Part I — Step Law](https://arxiv.org/abs/2503.04715)

通过大量模型与超参数网格搜索，拟合最优学习率、Batch Size 与模型规模、数据规模的关系。主要观察：最优 LR 随参数量和数据规模呈幂律变化；最优 Batch Size 主要随数据规模变化；固定模型与数据时，最优超参数附近存在相对平坦的 plateau；作者报告了跨模型形状、Dense/MoE 与数据分布的迁移结果。

论文报告约 3,700 个模型、约 100T token 和接近 100 万 H800 GPU-hours。贡献不是发现一个永远不变的 LR，而是建立可迁移的经验函数族。Loss 对规模呈幂律，并不意味着最优 LR 只由参数量决定：优化器、batch、训练时长、schedule 与结构都会改变训练动力学。

## 8. MoE Scaling：容量、计算与稀疏度

MoE 要区分总参数 \(N_{\mathrm{total}}\)、每 token 激活参数 \(N_{\mathrm{active}}\) 与激活比例/稀疏度。只用其中一个变量，可能把容量、计算量和优化动力学混为一谈。

- [Joint MoE Scaling Laws (2025)](https://arxiv.org/abs/2502.05172)：联合研究 MoE 规模变量与内存/计算效率。总容量可增长，而每 token 计算不必同比增长；Dense 参数量—算力关系不能不加修正地套用。
- [Towards Greater Leverage (2025)](https://arxiv.org/abs/2507.17702)：研究固定计算/内存预算下如何组合专家容量与每 token 计算。“激活参数相同”不保证不同 MoE 的有效计算或质量相同。
- [Hyperparameter Scaling Laws Across MoE Sparsity (2026)](https://arxiv.org/abs/2609.08690)：指出 ultra-sparse MoE 的最优 LR 和 Batch Size 会随 activation ratio 改变，且不能只靠总参数或激活参数解释。作者报告 1,800 次预训练实验、约 20T token，并在总参数 12B、激活比例 \(1/64\) 的 held-out MoE 上检验外推。

其经验形式为：

\[
\eta^*(C,A)=k_\eta C^{\gamma_\eta}A^{\delta_\eta},
\qquad
B^*(D,A)=k_B D^{\gamma_B}A^{\delta_B},
\]

其中 \(A\) 是 activation ratio。固定稀疏度时，最优 LR 更适合按训练算力 \(C\) 描述，最优 Batch Size 更适合按 token 数 \(D\) 描述；跨稀疏度时，\(A\) 作为额外乘法因子进入规律。**稀疏度是独立的超参数迁移维度，而不只是“激活参数更少”。**

## 9. Frontier pretraining：Scaling ladder 与 source scaling

- [MAI-Thinking-1 technical report](https://microsoft.ai/pdf/mai-thinking-1.pdf)
- [Frontier Pre-training at Microsoft AI (2026)](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/)

Microsoft AI 报告 MAI-Base-2 是 2.8T 总参数、80B 激活参数的 MoE，并使用 scaling ladder 预测最终表现。报告强调 **source scaling**：小规模实验若没有覆盖与目标模型相近的数据来源/分布，小模型上看似更好的架构或超参数，放大后可能过拟合或失效。报告使用 held-out evaluation 的 BPB—训练 FLOPs 曲线估计 recipe 的 FLOPs efficiency。

可信的 scaling ladder 不只是训练几个小模型再拟合 Loss；还应控制数据来源、训练配方、评估指标和架构变化，并通过 held-out scale 检验外推误差。

## 10. 总结对比

| 工作 | 主要问题 | 相对前作的推进 |
|:--|:--|:--|
| Kaplan (2020) | 参数、数据、算力与 Loss | 建立经验幂律与早期算力分配建议 |
| Chinchilla (2022) | 固定 FLOPs 下的 \(N,D\) | 用 IsoFLOP 修正模型过大、数据不足的分配 |
| Repeated Data / Data-Constrained | 重复曝光与数据上限 | 区分独立数据量与总训练 token |
| DeepSeek / replication | 真实 recipe 与可复现性 | 检验 scaling 的适用边界 |
| Optimal LR Across Horizons | LR 与 token horizon | 将训练时长纳入超参数设计 |
| Joint MoE / Greater Leverage | MoE 容量、激活计算与内存 | 从 Dense 推进到稀疏专家架构 |
| Step Law | LR、Batch Size 等超参数 | 通过网格搜索建立迁移规律 |
| Prescriptive Data-Constrained | 重复数据的过拟合代价 | 将数据约束纳入 compute-optimal 建议 |
| MoE Sparsity | 稀疏度与最优超参数 | 将 activation ratio 纳入超参数 scaling |
| MAI-Base-2 report | 真实前沿预训练 ladder | 强调 source scaling 与真实数据分布 |

## 11. 实验中最容易犯的错误

1. **把相关性当成定律**：报告拟合区间、误差与外推范围。
2. **只控制参数量，不控制算力**：模型更大但 token 更少，无法直接说明架构更好。
3. **把总 token 当成独立数据量**：重复曝光、数据质量与来源都会改变有效信息量。
4. **把总参数等同于计算量**：对 MoE 同时报告总参数、激活参数、稀疏度和实际 FLOPs。
5. **只报告初始 LR**：还要报告 token horizon 与 schedule。
6. **忽略 batch 的优化效率**：固定 token 数但改变 batch，会改变更新步数与优化噪声。
7. **把小模型最优点直接搬到大模型**：用 scaling ladder 并预留 held-out scale 检验外推。
8. **只看单一指标**：Loss、BPB 与下游能力不是同一个量，比较时必须使用可比评测。

## 12. 推荐阅读顺序

1. [Kaplan (2020)](https://arxiv.org/abs/2001.08361)：Loss scaling、训练动态与早期算力分配。
2. [Chinchilla (2022)](https://arxiv.org/abs/2203.15556)：IsoFLOP 与 compute-optimal 的 \(N,D\) 分配。
3. [Repeated Data](https://arxiv.org/abs/2205.10487) → [Scaling Data-Constrained](https://arxiv.org/abs/2305.16264)：独立数据与重复曝光。
4. [Scaling Optimal LR Across Token Horizons](https://arxiv.org/abs/2409.19913)：训练 horizon 对 LR 的影响。
5. [Step Law](https://arxiv.org/abs/2503.04715)：超参数迁移规律。
6. [Joint MoE Scaling](https://arxiv.org/abs/2502.05172) → [Towards Greater Leverage](https://arxiv.org/abs/2507.17702) → [MoE Sparsity](https://arxiv.org/abs/2609.08690)：容量、计算、稀疏度与优化。
7. [Prescriptive Data-Constrained](https://arxiv.org/abs/2605.01640) 与 [MAI frontier pretraining report](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/)：数据约束和前沿训练。

## 参考资料

- [Lilian Weng, Scaling Laws (2026)](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)
- [科学空间：幂律篇](https://kexue.fm/archives/11833)
- [Stanford CS336](https://stanford-cs336.github.io/) Lecture 09 / Lecture 11
- [MAI-Thinking-1 technical report](https://microsoft.ai/pdf/mai-thinking-1.pdf)
- [Frontier Pre-training at Microsoft AI (2026)](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/)
- [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)

---

*本笔记强调研究问题之间的关系，而不是把论文压缩成互不相关的摘要。具体模型的实验规模、公式定义与外推边界仍应回到原文核对。*