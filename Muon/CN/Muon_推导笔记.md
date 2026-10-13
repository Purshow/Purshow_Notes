# 从范数到优化器：SGD 与 Muon

## 1. 统一视角：在更新预算内降低损失

设当前参数矩阵为 $W\in\mathbb R^{m\times n}$，更新为同形矩阵 $\Delta W$，目标是最小化 $\mathcal L(W+\Delta W)$。若不限制 $\Delta W$，就等于一次求出最优参数矩阵，通常难以做到。因此，改为在当前参数附近逐步降低损失：每一步先给定更新预算 $\varepsilon>0$，再在预算内选择更新：

$$
\min_{\|\Delta W\|\leq\varepsilon}\mathcal L(W+\Delta W).
\tag{1}
$$

这里的范数用于衡量更新幅度，暂不指定。当预算足够小时，在当前参数处对损失作一阶展开：

$$
\mathcal L(W+\Delta W)\approx\mathcal L(W)+\langle G,\Delta W\rangle,
\qquad G=\nabla_W\mathcal L(W).
\tag{2}
$$

其中 $\langle G,\Delta W\rangle=\operatorname{Tr}(G^\top\Delta W)=\sum_{i,j}G_{ij}(\Delta W)_{ij}$。去掉常数 $\mathcal L(W)$，式（1）近似为：

$$
\min_{\|\Delta W\|\leq\varepsilon}\langle G,\Delta W\rangle.
\tag{3}
$$

为分开表示更新的方向和幅度，将非零更新写成：

$$
\Delta W
=\underbrace{\dfrac{\Delta W}{\|\Delta W\|}}_{\text{方向 }D}
\times\underbrace{\vphantom{\dfrac{\Delta W}{\|\Delta W\|}}\|\Delta W\|}_{\text{幅度 }\rho}
=\rho D.
\tag{4}
$$

其中 $\|D\|=1$，$\rho>0$ 是表示更新幅度的实数标量，预算约束为 $\rho\leq\varepsilon$。此时目标为 $\rho\langle G,D\rangle$。当 $G\ne0$ 时，最优方向的内积为负，沿该方向用满预算可使线性目标下降最多。因此，只需先求最优单位方向，再乘以预算 [1, 2]：

$$
D^*\in\arg\min_{\|D\|=1}\langle G,D\rangle,
\qquad
\boxed{\Delta W^*=\varepsilon D^*.}
\tag{5}
$$

以下推导考虑非零梯度；梯度为零时，局部问题可取零更新。

## 2. Frobenius 范数：从归一化更新到 SGD

Frobenius 范数 $\|D\|_{\mathrm F}=(\sum_{i,j}D_{ij}^2)^{1/2}$ 等于矩阵展平后的 $\ell_2$ 范数。选择 Frobenius 范数后，式（5）就是在 $\|D\|_{\mathrm F}=1$ 的约束下最小化 $\langle G,D\rangle$。由矩阵内积的柯西-施瓦茨不等式：

$$
|\langle G,D\rangle|
\leq\|G\|_{\mathrm F}\|D\|_{\mathrm F}
=\|G\|_{\mathrm F}.
\tag{6}
$$

因此，内积的下界为 $-\|G\|_{\mathrm F}$，在 $D=-G/\|G\|_{\mathrm F}$ 时取到。乘以预算，得到：

$$
D^*=-\frac{G}{\|G\|_{\mathrm F}},
\qquad
\boxed{\Delta W^*=-\varepsilon\frac{G}{\|G\|_{\mathrm F}}.}
\tag{7}
$$

固定预算 $\varepsilon$ 得到的是归一化梯度下降。普通 SGD 对应的预算为 $\varepsilon_t=\eta_t\|G_t\|_{\mathrm F}$，即学习率 $\eta_t$ 乘以当前小批量梯度矩阵 $G_t$ 的 Frobenius 范数。代入式（7）后，梯度范数相消，得到：

$$
\varepsilon_t=\eta_t\|G_t\|_{\mathrm F}
\quad\Longrightarrow\quad
\boxed{W_{t+1}=W_t-\eta_t G_t.}
\tag{8}
$$

这一更新可逐元素计算：$(\Delta W_t)_{ij}=-\eta_t G_{t,ij}$。它也是近端梯度（proximal gradient）在无非光滑项时的随机版本 [3]。

## 3. 固定谱范数预算得到 Muon

矩阵的另一种常用范数是谱范数，用来衡量矩阵对向量长度（$\ell_2$ 范数）的最大放大倍数，其值等于矩阵的最大奇异值：

$$
\|\Delta W\|_2=\max_{\|x\|_2=1}\|\Delta W x\|_2=\sigma_{\max}(\Delta W).
\tag{9}
$$

选择谱范数并令预算 $\varepsilon=\eta$ 后，式（3）就是在 $\|\Delta W\|_2\leq\eta$ 的约束下最小化 $\langle G,\Delta W\rangle$。为求出最优更新，先对梯度矩阵 $G$ 作紧致 SVD，仅保留非零奇异值：

$$
G=U_r\Sigma_rV_r^\top
=\sum_{i=1}^r\sigma_i u_iv_i^\top,\qquad \sigma_i>0.
\tag{10}
$$

由于 $u_i,v_i$ 都是单位向量，$u_i^\top\Delta W v_i\geq-\|\Delta W\|_2\geq-\eta$，所以：

$$
\langle G,\Delta W\rangle
=\operatorname{Tr}(G^\top\Delta W)
=\sum_{i=1}^r\sigma_i u_i^\top\Delta W v_i
\geq-\eta\sum_{i=1}^r\sigma_i.
\tag{11}
$$

取 $\Delta W=-\eta U_rV_r^\top$ 时，$\|\Delta W\|_2=\eta$ 且每个 $u_i^\top\Delta W v_i=-\eta$，恰好达到下界。因此，固定谱范数预算的一个最优更新为：

$$
\boxed{\Delta W^*=-\eta U_rV_r^\top.}
\tag{12}
$$

这直接给出 Muon 的核心更新 [4]。原奇异值 $\sigma_i$ 决定一阶损失最多能下降多少，即 $\eta\sum_i\sigma_i$；各非零奇异方向上的更新幅度则都为预算 $\eta$。相比之下，普通 SGD 在这些方向上的幅度为 $\eta\sigma_i$。实际 Muon 对动量矩阵进行近似正交化，并可配合维度缩放 [5]。

## 4. RMS→RMS 算子范数得到 $\mu$P 版 Muon

在线性层 $y=Wx$ 中，$n$ 为输入维度，$m$ 为输出维度。为比较不同宽度下激活分量的典型大小，使用均方根（RMS）：

$$
\|v\|_{\mathrm{RMS}}
=\sqrt{\frac{1}{d}\sum_{i=1}^d v_i^2}
=\frac{\|v\|_2}{\sqrt d},\qquad v\in\mathbb R^d.
\tag{13}
$$

输入和输出都用 RMS 衡量，便得到 RMS→RMS 算子范数 [4]：

$$
\begin{aligned}
\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}
&:=\max_{x\ne0}\frac{\|\Delta W x\|_{\mathrm{RMS}}}{\|x\|_{\mathrm{RMS}}}\\
&=\max_{x\ne0}\frac{\|\Delta W x\|_2/\sqrt m}{\|x\|_2/\sqrt n}
=\sqrt{\frac{n}{m}}\,\|\Delta W\|_2.
\end{aligned}
\tag{14}
$$

固定预算 $\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}\leq\eta$，等价于 $\|\Delta W\|_2\leq\eta\sqrt{m/n}$。因此，将式（12）的谱范数预算替换为 $\eta\sqrt{m/n}$，就得到 $\mu$P 版 Muon 更新 [4]：

$$
\boxed{\Delta W_{\mu\mathrm P}
=-\eta\sqrt{\frac{m}{n}}\,U_rV_r^\top.}
\tag{15}
$$

关于 $\mu$P 的具体内容，非常非常推荐参考这三篇文章哇 [6, 12, 13]（苏神和Greg Yang还是太权威了）.

## 5. 从 Muon 到 Moonlight

沿用前文记号，$\Delta W_t$ 表示梯度驱动的参数增量，**包含学习率及下降方向的负号**；权重衰减单独写出：

$$
\boxed{W_{t+1}=(1-\eta_t\lambda_{\mathrm{wd}})W_t+\Delta W_t.}
\tag{17}
$$

其中 $\lambda_{\mathrm{wd}}$ 是权重衰减系数，后文简称 WD。记 $H_t$ 为送入正交化的动量矩阵，普通动量与一种常见的 Nesterov 写法分别为：

$$
M_t=\mu M_{t-1}+G_t,\qquad
H_t=\begin{cases}
M_t,&\text{普通动量},\\
G_t+\mu M_t,&\text{Nesterov}.
\end{cases}
\tag{18}
$$

$$
\Delta W_t=-\eta_t\begin{cases}
\operatorname{NS}_5(H_t),&\text{原始 Muon 基线},\\
0.2\sqrt{\max(m,n)}\,\operatorname{NS}_5(H_t),&\text{Moonlight}.
\end{cases}
\tag{19}
$$

这里 $\operatorname{NS}_5$ 表示 5 步 Newton-Schulz 近似正交化，我们在之后再具体介绍。

**Moonlight + Kimi K2 引入的 QK-Clip，可概括为三点：**

1. **加入 weight decay。** 按式（17）施加解耦权重衰减，抑制权重及激活幅度持续增长。
2. **更新 RMS 对齐 Adam。** 乘以 $0.2\sqrt{\max(m,n)}$，使未乘学习率的更新 RMS 大致对齐 Adam 的典型量级 $0.2$ （此事在 https://kexue.fm/ 记载），从而可直接复用 Adam 调好的**学习率和权重衰减率**，减少重新搜索超参数的成本 [7, 8]。
3. **QK-Clip。** 在每步更新后，若某个注意力头的最大 attention logit 超过阈值，则按比例缩小 Q/K 权重，抑制 attention logit 爆炸 [9]。

这里的更新 RMS 与之前输入输出扰动比例的 RMS→RMS 算子范数不同， 指 $\operatorname{RMS}(\Delta W/\eta)=\|\Delta W\|_{\mathrm F}/(\eta\sqrt{mn})$，衡量更新元素的整体大小。

## 6. 核心对比表

除特别标注外，下表描述预训练配置。更新量沿用式（19）的写法，权重衰减按式（17）计算。

表中沿用式（19）的写法，$W\in\mathbb R^{m\times n}$ 表示实际参与一次更新计算的矩阵。按头处理时，$W$ 就是一个头对应的子矩阵，$m,n$ 取该子矩阵的行列数；各头分别计算更新后，按原位置组合成整个权重矩阵的更新。

$\operatorname{PE}_8$ 表示 8 步 Polar Express，$\operatorname{SK}_{11}$ 表示含弱行屏蔽的 11 步 Sinkhorn 均衡；$s$ 表示当前矩阵的更新缩放系数。Muown 中，$g$ 是行幅度，$H_{R,t}$ 是方向变量的动量矩阵，$\widehat m_{g,t},\widehat v_{g,t}$ 是幅度梯度的 Adam 矩估计。MiMo 公式按基础系数 $0.2$ 合并 RL 额外缩放 $0.5$，写为 $0.1$。具体计算见第 7 节，K3 的 QK-Clip 见式（27）。

| 模型 | 核心更新 | 相比原始 Muon 的主要修改 | 参数分工与设置 |
|:--|:--|:--|:--|
| **Qwen3.8-Flash-Next** | $\Delta W_t=-0.2\eta_t\sqrt{\max(m,n)}\,\operatorname{PE}_8(H_t)$。 | **Moonlight 缩放 + 8 步 Polar Express + 更细拆分**。QKV、GDN 输入投影按头拆；SwiGLU FC1 拆 gate/up，缩放按子矩阵尺寸计算。 | Nesterov，$\mu=0.95$。Embedding、LM head、router、GR 低秩投影用 AdamW；N-gram 表用 Adam。 |
| **DeepSeek-V4.1-Flash** | $\begin{aligned}[t]&\text{主干矩阵：}\\[-1pt]&\Delta W_t=-\eta_t s\,\operatorname{NS}(H_t),\\[1pt]&\text{Engram、embedding、LM head：}\\[-1pt]&\Delta W_t=-0.18\eta_t\sqrt n\,\operatorname{SK}_{11}(H_t).\end{aligned}$ | 在 V4 基础上增加 **Head-wise Muon**；**Engram 表、token embedding、LM head 改用 Sinkhorn 行列均衡更新**。 | Muon：目标 RMS 为 0.18，$\mu=0.95$、WD=0.1。Sinkhorn：**无 WD**；Engram LR 额外 ×5。Norm/非矩阵参数用 AdamW。 |
| **DeepSeek-V4-Pro** | $\begin{aligned}[t]\Delta W_t={}&-0.18\eta_t\sqrt{\max(m,n)}\\&\quad\cdot\operatorname{HybridNS}_{8+2}(H_t).\end{aligned}$ | **Moonlight 式缩放与 WD + 10 步混合 NS**：前 8 步快速逼近，后 2 步修正；RMS 缩放系数采用 **0.18**。 | Nesterov，$\mu=0.95$、WD=0.1。Embedding、LM head、RMSNorm、mHC 的静态 bias/gating factors 用 AdamW。 |
| **Kimi K3** | $\Delta W_t=-\eta_t s\,\operatorname{NS}(H_t)$。 | **Per-Head Muon + QK-Clip**：Q/K/V 按头更新，随后按各头最大 attention logit 裁剪 Q/K 权重。 | 矩阵参数采用 Muon，WD=0.1。**备注：** K3原文描述有点少，不确定有没有 RMS 对齐之类的。 |
| **MiMo-V2.6-Pro** | $\begin{aligned}[t]R_{t+1}&=R_t-0.1\eta_t\sqrt{\max(m,n)}\\&\quad\cdot\operatorname{NS}_{10}(H_{R,t}),\\[2pt]g_{t+1}&=g_t-\eta_t\frac{\widehat m_{g,t}}{\sqrt{\widehat v_{g,t}}+\epsilon},\\[2pt]\Delta W_{i,t}&=g_{i,t+1}\frac{R_{i,t+1}}{\|R_{i,t+1}\|_2}-W_{i,t}.\end{aligned}$ | **Muown：行方向与行幅度解耦优化。** 方向 $R$ 使用 Muon，行幅度 $g$ 使用 Adam，显式控制行范数，缓解权重谱范数漂移。 | **预训练：** AdamW。**Mid-training：** 隐藏层矩阵切换 Muown；embedding、LM head、router 保留 AdamW。**RL：** LR=$3\times10^{-6}$、WD=0、无 warmup、clip=1.0；Nesterov，$\mu=0.95$、NS=10、Muon 更新额外 ×0.5；Adam：$\beta_1=\beta_2=0.95$、$\epsilon=10^{-8}$；router 冻结。 |
| **Inkling** | $W_{t+1}=(1-\eta_t\lambda_t)W_t+\Delta W_t$。 | **Weight decay 与学习率平方耦合调度。** 每步实际衰减比例随 $\eta_t^2$ 调。 | **预训练：** 大矩阵使用 Muon，其余参数使用 Adam。 |
| **MAI-Thinking-1** | $\Delta W_t=-\eta_t\dfrac{\widehat m_t}{\sqrt{\widehat v_t}+\delta}$，逐元素计算。 | - | 预训练（MAI-Base-1）：$(\beta_1,\beta_2)=(0.95,0.925)$；WD 一般为 0.1，attention 为 0.01，embedding 为 0.005。RL：$\beta_1=\beta_2=0.95$，**WD=0**。 |

## 7. 关键改动的具体公式

### 7.1 正交化：NS与改进

我在这里只大概的说一下 NS 的做法，具体推荐非常推荐苏神的 blog（苏神还是太权威了 lol）。原始 Muon、DeepSeek-V4 与 Qwen 的共同计算形式为先归一化动量矩阵，再迭代矩阵多项式。以下用 $\delta$ 表示数值稳定常数（不使用时取 0），以区别前文的更新预算 $\varepsilon$：

$$
X_0=\frac{H_t}{\|H_t\|_{\mathrm F}+\delta}.
\tag{20}
$$

$$
\boxed{X_{k+1}=a_kX_k+b_k(X_kX_k^\top)X_k+c_k(X_kX_k^\top)^2X_k.}
\tag{21}
$$

原始 Muon / Moonlight 使用 5 步固定系数迭代：

$$
(a_k,b_k,c_k)=(3.4445,-4.7750,2.0315).
\tag{22}
$$

DeepSeek-V4-Pro 前 8 步使用式（22），后 2 步切换为修正系数使奇异值稳定在 1 附近：

$$
(a_k,b_k,c_k)=(2,-1.5,0.5).
\tag{23}
$$

Qwen3.8-Flash-Next 使用 **8 步 Polar Express [10] 的逐步系数调度**，并在 Frobenius 归一化中取 $\delta=10^{-14}$。

### 7.2 按头拆分：

Qwen 的拆分覆盖 QKV、GDN 输入投影以及 SwiGLU 的 gate/up；GLM-5 针对 MLA 的 Q/K/V 上投影，DeepSeek-V4.1 针对 Q/K，K3 则拆分 Q/K/V 的动量矩阵。

**K3 还保留了更新后的权重裁剪。** K2 QK-Clip 的通用形式为：当第 $h$ 个头的最大 attention logit $S_{\max}^h$ 超过阈值 $\tau$ 时，缩小 Q/K 权重：

$$
\begin{aligned}
r_h&=\begin{cases}1,&S_{\max}^h\leq\tau,\\
\tau/S_{\max}^h,&S_{\max}^h>\tau,\end{cases}\\
W_q^h&\leftarrow r_h^\alpha W_q^h,\qquad
W_k^h\leftarrow r_h^{1-\alpha}W_k^h.
\end{aligned}
\tag{27}
$$

其中 $\alpha$ 分配 Q/K 两侧的缩放。MLA 还需区分共享与非共享分量。

### 7.3 DeepSeek-V4.1：用 Sinkhorn 均衡行列尺度

他真的太爱 Sinkhorn 了...
对 Engram、embedding 或 LM head，设 $W\in\mathbb R^{m\times n}$，其中 $m$ 是矩阵行数（对 embedding 和 LM head 即词表大小），$n$ 是 hidden dimension。该组采用以下 Nesterov 动量写法：

$$
M_t=0.95M_{t-1}+0.05G_t,\qquad
H_t=0.95M_t+0.05G_t.
\tag{28}
$$

先将 $H_t$ 中行范数不超过平均行范数 $10^{-3}$ 倍的行置零，得到 $U^{(0)}$；再交替进行行、列归一化：

$$
U^{(k)}=\begin{cases}
\operatorname{RowL2Norm}_{\delta}(U^{(k-1)}),&k\text{ 为奇数},\\
\operatorname{ColL2Norm}_{\delta}(U^{(k-1)}),&k\text{ 为偶数},
\end{cases}
\quad k=1,\ldots,11.
\tag{29}
$$

每次除以对应行或列的 $\ell_2$ 范数加 $\delta=10^{-20}$。最后一步为行归一化，乘以 $\sqrt n$ 后将单位行 $\ell_2$ 范数转换为单位行 RMS，再用 $0.18$ 对齐更新量级：

$$
\boxed{W_{t+1}=W_t-0.18\eta_t\sqrt n\,U^{(11)}.}
\tag{30}
$$

这一参数组没有 weight decay，Engram 的学习率额外乘以 5。Sinkhorn 通过行列缩放均衡更新幅度，不保证奇异值相等，也不进行 Muon 式正交化。其目标是近似均衡行列 RMS；被置零的行仍保持零。

### 7.4 MiMo：Muown 单独学习权重行幅度

Muown [11] 是认为 Muon 没有独立控制权重自身的幅度可能会导致权重的行范数持续增长将，因此，它将权重矩阵的每一行表示为行幅度与归一化方向的乘积：

$$
W_i=g_i\frac{R_i}{\|R_i\|_2}.
\tag{31}
$$

其中 $g_i$ 是可学习的行幅度。令 $r_i=\|R_i\|_2$、$d_i=R_i/r_i$，给定权重梯度 $G_i$，按链式法则分离出幅度梯度 $q_i$ 和方向变量梯度 $F_i$：

$$
q_i=\langle G_i,d_i\rangle,\qquad
F_i=\frac{g_i}{r_i}(G_i-q_id_i).
\tag{32}
$$

用 Adam 更新 $g$，用 Muon 更新 $R$。省略权重衰减项，更新可写为：

$$
\begin{aligned}
M_t^R&=\mu M_{t-1}^R+F_t,\\
R^+&=R-\eta_ta_{m,n}\operatorname{NS}(F_t+\mu M_t^R),\\
g^+&=g-\eta_t\frac{\widehat m_g}{\sqrt{\widehat v_g}+\epsilon}.
\end{aligned}
\tag{33}
$$

其中 $\widehat m_g,\widehat v_g$ 是 $q$ 的 Adam 矩估计，$g$ 的更新逐元素计算,$a_{m,n}=0.2\sqrt{\max(m,n)}$。再重构实际权重：

$$
\boxed{W_i^+=g_i^+\frac{R_i^+}{\|R_i^+\|_2}.}
\tag{34}
$$


**参考资料**

[1] 苏剑林：[Muon 优化器赏析：从向量到矩阵的本质跨越](https://kexue.fm/archives/10592)。

[2] Bernstein & Newhouse：[Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325v2)。

[3] Parikh & Boyd：[Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)。

[4] Jeremy Bernstein：[Deriving Muon](https://jeremybernste.in/writing/deriving-muon)。

[5] Keller Jordan：[Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)。

[6] Yang, Simon & Bernstein：[A Spectral Condition for Feature Learning](https://papers.cool/arxiv/2310.17813)。

[7] 苏剑林：[Muon续集：为什么我们选择尝试Muon？](https://kexue.fm/archives/10739)。

[8] 苏剑林：[为什么Adam的Update RMS是0.2？](https://kexue.fm/archives/11267)。

[9] 苏剑林：[QK-Clip：让Muon在Scaleup之路上更进一步](https://kexue.fm/archives/11126)。

[10] Amsel, Persson, Musco & Gower：[The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm](https://arxiv.org/abs/2505.16932)。

[11] Lion, Hübler, Li, Orvieto & He：[Muown: Row-Norm Control for Muon Optimization](https://arxiv.org/abs/2605.10797v1)。

[12] 苏剑林：[高阶MuP：更简明但更高明的谱条件缩放](https://kexue.fm/archives/10795)。

[13] 苏剑林：[MuP之上：2. 线性层与最速下降](https://kexue.fm/archives/11605)。
