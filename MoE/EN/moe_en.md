# MoE

`R` = routed expert, `S` = always-active shared expert, and `Dense×d` = the first $d$ layers use a standard dense FFN.

## Three Main Load-Balancing Methods

Let $N$ be the number of experts, $K$ the Top-$K$ value, and $T$ the number of tokens. Define the actual load ratio $f_i$ and routing-probability ratio $P_i$ for expert $i$ as

$$
f_i = \frac{1}{KT}\sum_{t=1}^{T}\mathbf{1}[i\in\mathcal{S}_t],
\qquad
P_i = \frac{1}{T}\sum_{t=1}^{T}p_{t,i}.
$$

### A. Traditional GShard/Switch Auxiliary Loss

$$
\boxed{
\mathcal{L}_{\text{bal}}
 = \alpha N\sum_{i=1}^{N}f_iP_i
}
$$

This directly changes the Router through gradient updates.

### B. DeepSeek-Style Auxiliary-Loss-Free Balancing

$$
\mathcal{S}_t = \operatorname{TopK}_i(s_{t,i}+b_i).
$$

$$
w_{t,i} = c\frac{s_{t,i}}
{\sum_{j\in\mathcal{S}_t}s_{t,j}},
\qquad i\in\mathcal{S}_t.
$$

The bias $b_i$ **only affects expert selection, not the final mixing weights**. It is updated from load statistics. Here $f_i$ is the actual load ratio of expert $i$; the target load is uniform across experts:
$\bar{\mathbf f}=(1/N,\ldots,1/N)$, i.e. $\bar f_i=1/N$.
Underloaded experts receive a positive bias correction, while overloaded experts receive a negative one:

$$
\boxed{
b_i\leftarrow b_i+\gamma\operatorname{sign}(\bar f_i-f_i)
}
$$

### C. Kimi K3 Quantile Balancing

Kimi K3 does not use an additional load-balancing loss. Instead, it estimates the next-step expert bias directly from the Router-score margin distribution.

For each token $i$, compute the current biased scores $s_{i,j}+b_j^{(t)}$ and take Top-$(K+1)$. Let the $(K+1)$-th largest value be the biased cutoff:

$$
\alpha_i^{(t)}
= \operatorname{Top}_{K+1,j}\left(s_{i,j}+b_j^{(t)}\right).
$$

Then compute the margin of each expert $j$ relative to this cutoff using the original Router score:

$$
m_{i,j}^{(t)} = s_{i,j}-\alpha_i^{(t)}.
$$

The next-step, uncentered expert bias is

$$
\boxed{
\hat b_j^{(t+1)}
= -\operatorname{Quantile}_{1-K/N}
\left(m_{:,j}^{(t)}\right)
}
$$

Finally, K3 also mean-centers the bias across experts. 

$$
\boxed{
b^{(t+1)}
= \hat b^{(t+1)}
- \operatorname{mean}\left(\hat b^{(t+1)}\right)\mathbf{1}
}
$$


## Model Comparison

| Model | Total / active parameters | MoE configuration | Router & weights | Load balancing | Dense layers | Special design |
|:---|---:|:---|:---|:---|:---|:---|
| Kimi K3 | 2.78T / 104.2B | 93 layers; $16/896R + 2S$ | $s=\operatorname{Sigmoid}(z)$; select Top-16 by $s+b$; normalize $s$ within Top-16; scale $=1.0$ | Class C | First 1 layer | Stable LatentMoE: R uses the full input for routing, then compresses the input into a latent representation for weighted aggregation by Top-16 SiTU-GLU experts, followed by RMSNorm, up-projection, and addition with the Shared Expert output. |
| DeepSeek-V4-Pro | 1.6T / 49B | 61 layers; $6/384R + 1S$ | $s=\sqrt{\operatorname{softplus}(z)}$; select Top-6 by $s+b$; normalize $s$ within Top-6; scale $=2.5$ | Class B; adds a lightweight sequence-wise loss | None; first three layers use Hash-MoE | The first three layers use Hash-MoE, where the expert is determined by a predefined hash function over the input Token ID; clamped SwiGLU. |
| GLM-5.2 | 744B / 40B | 78 layers; $8/256R + 1S$ | Sigmoid; select Top-8 by $s+b$; normalize within Top-8; scale $=2.5$ | Class B | First 3 layers | No additional design |
| MAI-Thinking-1 | 962B / 34.7B | 78 layers; $8/512R$ | Softmax gating; select Top-8; whether the selected weights are renormalized and the final scale are unspecified | Class A | Alternating | LatentMoE: routes using the full representation while transmitting only the compressed latent. |
| TML Inkling | 975B / 41B | 66 layers; $6/256R + 2S$ | Sigmoid; add bias only to Routed Experts before selecting Top-6; jointly normalize the 6R and 2S scores; apply scale $=8\times$ a learnable per-layer scale to the normalized joint aggregate | Class B | First 2 layers | The Sigmoid scores of 6 Routed Experts and 2 Shared Experts are jointly normalized; the joint aggregate receives a fixed factor of 8 and a learnable per-layer scale (this value is particularly interesting—worth inspecting). |
| MiMo-V2.5-Pro | 1.02T / 42B | 70 layers; $8/384R$ | Sigmoid; select Top-8 by $s+b$; normalize $s$ within Top-8 | Class B | First 1 layer | No additional design |
| MiniMax-M3 | 428B / 23B | 60 layers; $4/128R + 1S$ | Sigmoid; select Top-4 by $s+b$; normalize the original $s$ within Top-4; scale $=2.0$ | Class B | First 3 layers | Clamped SwiGLU. |
| Qwen3.5-397B-A17B | 397B / 17B | 60 layers; $10/512R + 1S$ | Full 512-expert Softmax; select Top-10 by $p$; renormalize within Top-10; scale $=1.0$; the Shared Expert has a separate Sigmoid Gate | Class A | None | The Shared Expert is scored separately. |
| LongCat-2.0 | 1.6T / 48B | 38 layers; $12/(768R + 128Z)$, where $Z$ denotes a Zero-compute Identity Expert | Full 896-expert Softmax; select Top-12 by $p+b$; no Top-12 renormalization; scale $=9$ | Class B; adds a lightweight auxiliary loss adapted to Zero Experts | None | The 128 $Z$ experts are zero-compute Identity Experts (MoE++—great!). |
| HY-3 | 295B / 21B | 80 layers; $8/192R + 1S$ | Sigmoid + expert bias; select Top-8 by $s+b$; normalize within Top-8; scale $=2.826$ | Class B | First 1 layer | No additional design |
| DeepSeek-V3 | 671B / 37B | 61 layers; $8/256R + 1S$ | Sigmoid; select Top-8 by $s+b$; normalize $s$ within Top-8; scale $=2.5$ | Class B; adds a lightweight sequence-wise loss | First 3 layers | $256R\rightarrow 8$ groups; select Top-4 groups; then select Top-8 experts within them; each token visits at most 4 nodes. |
