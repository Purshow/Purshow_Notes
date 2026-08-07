# MoE

`R`＝路由专家，`S`＝始终激活的共享专家，`Dense×d`＝前 $d$ 层使用普通 Dense FFN。

## 三种主要负载均衡方法

令 $N$ 为专家数，$K$ 为 Top-$K$，$T$ 为 Token 数。定义专家 $i$ 的实际负载比例 $f_i$ 和路由概率比例 $P_i$：

$$
f_i = \frac{1}{KT}\sum_{t=1}^{T}\mathbf{1}[i\in\mathcal{S}_t],
\qquad
P_i = \frac{1}{T}\sum_{t=1}^{T}p_{t,i}.
$$

### A．传统 GShard/Switch 辅助损失

$$
\boxed{
\mathcal{L}_{\text{bal}}
 = \alpha N\sum_{i=1}^{N}f_iP_i
}
$$

它会通过梯度直接改变 Router。

### B．DeepSeek 式 Auxiliary-Loss-Free

$$
\mathcal{S}_t = \operatorname{TopK}_i(s_{t,i}+b_i).
$$

$$
w_{t,i} = c\frac{s_{t,i}}
{\sum_{j\in\mathcal{S}_t}s_{t,j}},
\qquad i\in\mathcal{S}_t.
$$

其中 $b_i$ **只影响选谁，不影响最终混合权重**，根据负载统计更新：

其中 $f_i$ 是专家 $i$ 的实际负载比例；目标负载设为专家上的均匀分布：
$\bar{\mathbf f}=(1/N,\ldots,1/N)$，即 $\bar f_i=1/N$。
因此，负载偏低的专家会得到正向 bias 修正，负载偏高的专家会得到负向修正。

$$
\boxed{
b_i\leftarrow b_i+\gamma\operatorname{sign}(\bar f_i-f_i)
}
$$

### C．Kimi K3 Quantile Balancing

不使用额外的负载均衡 loss，而是根据当前训练步的 Router score margin 分布直接估计下一步的专家偏置。

对每个 Token $i$，先在当前带偏置分数 $s_{i,j}+b_j^{(t)}$ 上取 Top-$(K+1)$，将第 $(K+1)$ 大值记为带偏置 cutoff：

$$
\alpha_i^{(t)}
= \operatorname{Top}_{K+1,j}\left(s_{i,j}+b_j^{(t)}\right).
$$

随后对每个专家 $j$ 计算原始 Router score 相对该 cutoff 的 margin：

$$
m_{i,j}^{(t)} = s_{i,j}-\alpha_i^{(t)}.
$$

并令下一步的未中心化专家偏置为：

$$
\boxed{
\hat b_j^{(t+1)}
= -\operatorname{Quantile}_{1-K/N}
\left(m_{:,j}^{(t)}\right)
}
$$

最后K3还对所有专家 bias 做均值中心化（这一步其实不影响排序，问了下苏神是因为这么做好看哈哈哈哈）：

$$
\boxed{
b^{(t+1)}
= \hat b^{(t+1)}
- \operatorname{mean}\left(\hat b^{(t+1)}\right)\mathbf{1}
}
$$


## 模型对比

| 模型 | 总参数 / 激活参数 | MoE 配置 | Router 与权重 | 负载均衡 | Dense 部分 | 特殊设计 |
|:---|---:|:---|:---|:---|:---|:---|
| Kimi K3 | 2.78T / 104.2B | 93 层；$16/896R + 2S$ | $s=\operatorname{Sigmoid}(z)$；按 $s+b$ 选 Top-16；Top-16 内 $s$ 归一化；scale $=1.0$ | C 类 | 前 1 层 | Stable LatentMoE：R 用完整输入做 Router，同时将输入降维到 latent，送入 Top-16 SiTU-GLU 专家加权聚合，再 RMSNorm、升维，并与 Shared Expert 输出相加。 |
| DeepSeek-V4-Pro | 1.6T / 49B | 61 层；$6/384R + 1S$ | $s=\sqrt{\operatorname{softplus}(z)}$；按 $s+b$ 选 Top-6；Top-6 内 $s$ 归一化；scale $=2.5$ | B 类；加轻量 sequence-wise loss | 没有，前三层Hash-MoE | 前三层使用 Hash-MoE，由输入 Token ID 的预定义哈希函数确定 expert；clamped SwiGLU。 |
| GLM-5.2 | 744B / 40B | 78 层；$8/256R + 1S$ | Sigmoid；按 $s+b$ 选 Top-8；Top-8 内归一化；scale $=2.5$ | B 类 | 前 3 层 | 无额外设计 |
| MAI-Thinking-1 | 962B / 34.7B | 78 层；$8/512R$ | Softmax gating；取 Top-8；Top-8 后是否再次归一化未说明；scale 未说明 | A 类 | 交替 | LatentMoE：用完整表示路由，只传输压缩 latent。 |
| TML Inkling | 975B / 41B | 66 层；$6/256R + 2S$ | Sigmoid；仅 Routed Experts 加 bias 后选 Top-6；6R 与 2S 分数联合归一化；归一化后的联合加权结果再乘 scale $=8\times$ 每层可学习 scale | B 类 | 前 2 层 | 6 个 Routed Expert 和 2 个 Shared Expert 的 Sigmoid 分数联合归一化；联合加权结果统一施加固定常数 8 和每层可学习 scale（这个值特别有意思，有兴趣的可以print下）。 |
| MiMo-V2.5-Pro | 1.02T / 42B | 70 层；$8/384R$ | Sigmoid；按 $s+b$ 选 Top-8；Top-8 内 $s$ 归一化 | B 类 | 前 1 层 | 无额外设计 |
| MiniMax-M3 | 428B / 23B | 60 层；$4/128R + 1S$ | Sigmoid；按 $s+b$ 选 Top-4；Top-4 内原始 $s$ 归一化；scale $=2.0$ | B 类 | 前 3 层 | clamped SwiGLU。 |
| Qwen3.5-397B-A17B | 397B / 17B | 60 层；$10/512R + 1S$ | 全 512 Softmax；按 $p$ 取 Top-10；Top-10 内再次归一化；scale $=1.0$；Shared Expert 另有 Sigmoid Gate | A 类 | 没有 | 共享专家单独打分。 |
| LongCat-2.0 | 1.6T / 48B | 38 层；$12/(768R + 128Z)$，其中 $Z$ 为 Zero-compute Identity Expert | 全 896 Softmax；按 $p+b$ 取 Top-12；不做 Top-12 renorm；scale $=9$ | B 类；加轻量 Aux loss，并适配 Zero Expert | 没有 | 128 个 $Z$ Expert 是零计算 Identity Expert（MoE++伟大！）。 |
| HY-3 | 295B / 21B | 80 层；$8/192R + 1S$ | Sigmoid + expert bias；按 $s+b$ 选 Top-8；Top-8 内归一化；scale $=2.826$ | B 类 | 前 1 层 | 无额外设计 |
| DeepSeek-V3 | 671B / 37B | 61 层；$8/256R + 1S$ | Sigmoid；按 $s+b$ 选 Top-8；Top-8 内 $s$ 归一化；scale $=2.5$ | B 类；加轻量 sequence-wise loss | 前 3 层 | $256R\rightarrow 8$ 个 group；选 Top-4 group；再在其中选 Top-8 expert；每个 Token 最多访问 4 个节点。 |
