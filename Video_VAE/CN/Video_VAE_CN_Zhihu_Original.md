# 视频生成模型 VAE

## 1. 压缩参数与 DiT token 数

倍率按**时间 × 高度 × 宽度**排列，<img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 为 VAE 通道数。末列为 **DiT patchify 后、投影前**的形状，patch 内元素并入通道维。

<img src="https://www.zhihu.com/equation?tex=%0AR_%7B%5Cmathrm%7Bnom%7D%7D%3D%5Cfrac%7B3r_t%20r_h%20r_w%7D%7BC%7D%2C%5Cqquad%0AR_%7B%5Cmathrm%7Bactual%7D%7D%3D%5Cfrac%7B3THW%7D%7BC%5C%2CT_zH_zW_z%7D%2C%5Cqquad%0AN%3D%5Cfrac%7BT_zH_zW_z%7D%7Bp_tp_hp_w%7D.%0A" alt="&#10;R_{\mathrm{nom}}=\frac{3r_t r_h r_w}{C},\qquad&#10;R_{\mathrm{actual}}=\frac{3THW}{C\,T_zH_zW_z},\qquad&#10;N=\frac{T_zH_zW_z}{p_tp_hp_w}.&#10;" class="ee_img tr_noresize" eeimg="1">

DiT patch 为 <img src="https://www.zhihu.com/equation?tex=p_t%5Ctimes%20p_h%5Ctimes%20p_w" alt="p_t\times p_h\times p_w" class="ee_img tr_noresize" eeimg="1">，只重排元素；压缩比按 VAE 输出计算，已计入 VAE 内部 patchify。输入尺寸须对齐。

<table>
<thead>
<tr>
<th>模型</th>
<th>VAE</th>
<th>时间 × 高 × 宽压缩</th>
<th>Latent<br>通道 <img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"></th>
<th>元素压缩比 <img src="https://www.zhihu.com/equation?tex=R_%7B%5Cmathrm%7Bnom%7D%7D" alt="R_{\mathrm{nom}}" class="ee_img tr_noresize" eeimg="1"></th>
<th>DiT patchify</th>
<th>DiT 输入形状</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>MiniMax H3</strong></td>
<td><strong>H3-VisualVAE</strong><br>因果 3D CNN encoder + <strong>非因果 ViT decoder</strong>；一次投影恢复时空像素块。</td>
<td><strong><img src="https://www.zhihu.com/equation?tex=4%5Ctimes16%5Ctimes16" alt="4\times16\times16" class="ee_img tr_noresize" eeimg="1"></strong></td>
<td><strong>24</strong></td>
<td><strong>128:1</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=1%5Ctimes2%5Ctimes2" alt="1\times2\times2" class="ee_img tr_noresize" eeimg="1"></td>
<td><img src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C96%2CT_%7B%5Cmathrm%7BH3%7D%7D%2C%5Cfrac%7BH%7D%7B32%7D%2C%5Cfrac%7BW%7D%7B32%7D%5Cright%5D" alt="\left[B,96,T_{\mathrm{H3}},\frac{H}{32},\frac{W}{32}\right]" class="ee_img tr_noresize" eeimg="1"></td>
</tr>
<tr>
<td><strong>LTX-2.5</strong></td>
<td><strong>LTX-2.5 Video VAE</strong><br>因果 3D CNN encoder + <strong>非因果 3D CNN / NA Transformer 扩散 decoder</strong>（CNN 按默认配置）；空间 patchify ×4；扩散版单步像素去噪。</td>
<td><strong><img src="https://www.zhihu.com/equation?tex=8%5Ctimes32%5Ctimes32" alt="8\times32\times32" class="ee_img tr_noresize" eeimg="1"></strong></td>
<td><strong>128</strong></td>
<td><strong>192:1</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=1%5Ctimes1%5Ctimes1" alt="1\times1\times1" class="ee_img tr_noresize" eeimg="1"></td>
<td><img src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C128%2C1%2B%5Cfrac%7BT-1%7D%7B8%7D%2C%5Cfrac%7BH%7D%7B32%7D%2C%5Cfrac%7BW%7D%7B32%7D%5Cright%5D" alt="\left[B,128,1+\frac{T-1}{8},\frac{H}{32},\frac{W}{32}\right]" class="ee_img tr_noresize" eeimg="1"></td>
</tr>
<tr>
<td><strong>FLUX 3 Action</strong></td>
<td><strong>FLUX 3 Video VAE</strong><br>因果 NA Transformer encoder + <strong>非因果 NA Transformer decoder</strong>；<code>5×5×5</code> 邻域 attention。</td>
<td><strong><img src="https://www.zhihu.com/equation?tex=4%5Ctimes32%5Ctimes32" alt="4\times32\times32" class="ee_img tr_noresize" eeimg="1"></strong></td>
<td><strong>96</strong></td>
<td><strong>128:1</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=1%5Ctimes1%5Ctimes1" alt="1\times1\times1" class="ee_img tr_noresize" eeimg="1"></td>
<td><img src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C96%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B32%7D%2C%5Cfrac%7BW%7D%7B32%7D%5Cright%5D" alt="\left[B,96,1+\frac{T-1}{4},\frac{H}{32},\frac{W}{32}\right]" class="ee_img tr_noresize" eeimg="1"></td>
</tr>
<tr>
<td><strong>Wan 2.1 / 2.2 A14B / Lingbot-Video</strong></td>
<td><strong>Wan2.1-VAE</strong><br>因果 3D CNN encoder + 因果 3D CNN decoder；帧内空间 attention、逐层历史缓存。</td>
<td><strong><img src="https://www.zhihu.com/equation?tex=4%5Ctimes8%5Ctimes8" alt="4\times8\times8" class="ee_img tr_noresize" eeimg="1"></strong></td>
<td><strong>16</strong></td>
<td><strong>48:1</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=1%5Ctimes2%5Ctimes2" alt="1\times2\times2" class="ee_img tr_noresize" eeimg="1"></td>
<td><img src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C64%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B16%7D%2C%5Cfrac%7BW%7D%7B16%7D%5Cright%5D" alt="\left[B,64,1+\frac{T-1}{4},\frac{H}{16},\frac{W}{16}\right]" class="ee_img tr_noresize" eeimg="1"></td>
</tr>
<tr>
<td><strong>Wan 2.2 TI2V-5B</strong></td>
<td><strong>Wan2.2-VAE</strong><br>因果 3D CNN encoder + 因果 3D CNN decoder；空间 patchify ×2、残差采样 shortcut。</td>
<td><strong><img src="https://www.zhihu.com/equation?tex=4%5Ctimes16%5Ctimes16" alt="4\times16\times16" class="ee_img tr_noresize" eeimg="1"></strong></td>
<td><strong>48</strong></td>
<td><strong>64:1</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=1%5Ctimes2%5Ctimes2" alt="1\times2\times2" class="ee_img tr_noresize" eeimg="1"></td>
<td><img src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C192%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B32%7D%2C%5Cfrac%7BW%7D%7B32%7D%5Cright%5D" alt="\left[B,192,1+\frac{T-1}{4},\frac{H}{32},\frac{W}{32}\right]" class="ee_img tr_noresize" eeimg="1"></td>
</tr>
<tr>
<td><strong>HunyuanVideo-1.5</strong></td>
<td><strong>AutoencoderKLConv3D</strong><br>因果 3D CNN encoder + 因果 3D CNN decoder；瓶颈因果 attention、残差式 3D pixel shuffle。</td>
<td><strong><img src="https://www.zhihu.com/equation?tex=4%5Ctimes16%5Ctimes16" alt="4\times16\times16" class="ee_img tr_noresize" eeimg="1"></strong></td>
<td><strong>32</strong></td>
<td><strong>96:1</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=1%5Ctimes1%5Ctimes1" alt="1\times1\times1" class="ee_img tr_noresize" eeimg="1"></td>
<td><img src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C32%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B16%7D%2C%5Cfrac%7BW%7D%7B16%7D%5Cright%5D" alt="\left[B,32,1+\frac{T-1}{4},\frac{H}{16},\frac{W}{16}\right]" class="ee_img tr_noresize" eeimg="1"></td>
</tr>
<tr>
<td><strong>MAGI-2 Preview</strong></td>
<td><strong>Wan2.2-VAE + TurboVAED</strong><br>因果 3D CNN encoder + <strong>非因果 3D CNN 蒸馏 decoder</strong>。</td>
<td><strong><img src="https://www.zhihu.com/equation?tex=4%5Ctimes16%5Ctimes16" alt="4\times16\times16" class="ee_img tr_noresize" eeimg="1"></strong></td>
<td><strong>48</strong></td>
<td><strong>64:1</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=1%5Ctimes1%5Ctimes1" alt="1\times1\times1" class="ee_img tr_noresize" eeimg="1"></td>
<td><img src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C48%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B16%7D%2C%5Cfrac%7BW%7D%7B16%7D%5Cright%5D" alt="\left[B,48,1+\frac{T-1}{4},\frac{H}{16},\frac{W}{16}\right]" class="ee_img tr_noresize" eeimg="1"></td>
</tr>
</tbody>
</table>

**H3 的时间压缩比较特殊，严格来说并不是 4:1，元素压缩比也要比 128:1 小一些。**

## 2. Wan2.1-VAE 

接着我们具体介绍一下应该是最经典的 VAE：Wan2.1-VAE。以 **81 帧、480×832** 的 RGB 视频为例：

**整段视频：**`[3,81,480,832]` → **latent：**`[16,21,60,104]` → **重建视频：**`[3,81,480,832]`。

<img src="https://www.zhihu.com/equation?tex=%0AT_z%3D1%2B%5Cfrac%7B81-1%7D%7B4%7D%3D21%2C%5Cqquad%20H_z%3D%5Cfrac%7B480%7D%7B8%7D%3D60%2C%5Cqquad%20W_z%3D%5Cfrac%7B832%7D%7B8%7D%3D104.%0A" alt="&#10;T_z=1+\frac{81-1}{4}=21,\qquad H_z=\frac{480}{8}=60,\qquad W_z=\frac{832}{8}=104.&#10;" class="ee_img tr_noresize" eeimg="1">

### 2.1 Encoder：跟踪后续的 4 帧视频块 

整段视频按 **1、4、4……帧**分成 **21 个块**：1 个首帧块，加上 20 个后续块，Wan2.1-VAE 会做 Feature Cache，保留前面块的历史特征。下面我们用一个后续块来看看 Wan2.1-VAE 的具体 forward 过程：

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>当前视频块</td>
<td><code>[3,4,480,832]</code></td>
</tr>
<tr>
<td>入口</td>
<td>因果卷积 <strong>3→96</strong></td>
<td><code>[96,4,480,832]</code></td>
</tr>
<tr>
<td>第一级</td>
<td>残差块 ×2，输出 96 通道 → <strong>空间下采样</strong></td>
<td><code>[96,4,240,416]</code></td>
</tr>
<tr>
<td>第二级</td>
<td>残差块 ×2，输出 192 通道 → <strong>时空下采样</strong></td>
<td><code>[192,2,120,208]</code></td>
</tr>
<tr>
<td>第三级</td>
<td>残差块 ×2，输出 384 通道 → <strong>时空下采样</strong></td>
<td><code>[384,1,60,104]</code></td>
</tr>
<tr>
<td>第四级</td>
<td>残差块 ×2，输出 384 通道</td>
<td><code>[384,1,60,104]</code></td>
</tr>
<tr>
<td>瓶颈</td>
<td>残差块 → 空间 Attention → 残差块</td>
<td><code>[384,1,60,104]</code></td>
</tr>
<tr>
<td>出口</td>
<td>RMSNorm → SiLU → 因果卷积 <strong>384→32</strong></td>
<td><code>[32,1,60,104]</code></td>
</tr>
</tbody>
</table>

全部 21 个块编码完成后，先沿时间维拼接，经过一个 `1×1×1` 卷积，再拆分为 <img src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" class="ee_img tr_noresize" eeimg="1"> 和 `log_var`，各为 `[16,21,60,104]`，对 <img src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" class="ee_img tr_noresize" eeimg="1"> 按通道标准化后作为输入。

### 2.2 Decoder：跟踪后续的 latent 时间步 

先对完整 latent **撤销标准化**，经过 **`1×1×1` 卷积，16→16**，形状仍为 `[16,21,60,104]`。随后逐个 latent 时间位置解码；下表跟踪一个后续时间步：

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>撤销标准化后的 latent 块</td>
<td><code>[16,1,60,104]</code></td>
</tr>
<tr>
<td>入口</td>
<td>因果卷积 <strong>16→384</strong></td>
<td><code>[384,1,60,104]</code></td>
</tr>
<tr>
<td>瓶颈</td>
<td>残差块 → 空间 Attention → 残差块</td>
<td><code>[384,1,60,104]</code></td>
</tr>
<tr>
<td>第一级</td>
<td>残差块 ×3，输出 384 通道 → <strong>时空上采样，384→192</strong></td>
<td><code>[192,2,120,208]</code></td>
</tr>
<tr>
<td>第二级</td>
<td>残差块 ×3，先将 <strong>192 升至 384</strong> → <strong>时空上采样，384→192</strong></td>
<td><code>[192,4,240,416]</code></td>
</tr>
<tr>
<td>第三级</td>
<td>残差块 ×3，输出 192 通道 → <strong>空间上采样，192→96</strong></td>
<td><code>[96,4,480,832]</code></td>
</tr>
<tr>
<td>第四级</td>
<td>残差块 ×3，输出 96 通道</td>
<td><code>[96,4,480,832]</code></td>
</tr>
<tr>
<td>出口</td>
<td>RMSNorm → SiLU → 因果卷积 <strong>96→3</strong></td>
<td><code>[3,4,480,832]</code></td>
</tr>
</tbody>
</table>

其中，首帧会跳过时间重采样，只输出 **1 帧**。

### 2.3 上下采样 

<table>
<thead>
<tr>
<th>操作</th>
<th>具体实现</th>
<th>形状变化</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>空间下采样</strong></td>
<td>逐帧右侧、下侧补 1 → <code>3×3 Conv2d</code>，<code>stride=2</code></td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各减半；<img src="https://www.zhihu.com/equation?tex=C%2CT" alt="C,T" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>时间下采样</strong></td>
<td>拼接 <strong>1 帧历史缓存</strong> → <code>3×1×1</code> 时间卷积，时间步长 2</td>
<td>后续块 <img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 减半；<img src="https://www.zhihu.com/equation?tex=C%2CH%2CW" alt="C,H,W" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>空间上采样</strong></td>
<td>逐帧最近邻插值 ×2（<code>nearest-exact</code>）→ <code>3×3 Conv2d</code>，<img src="https://www.zhihu.com/equation?tex=C%5Crightarrow%20C%2F2" alt="C -> C/2" class="ee_img tr_noresize" eeimg="1"></td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各翻倍，<img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 减半；<img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>时间上采样</strong></td>
<td><code>3×1×1</code> 因果卷积，<img src="https://www.zhihu.com/equation?tex=C%5Crightarrow2C" alt="C ->2C" class="ee_img tr_noresize" eeimg="1"> → 将两组通道重排到时间维</td>
<td>后续块 <img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 翻倍；最终 <img src="https://www.zhihu.com/equation?tex=C%2CH%2CW" alt="C,H,W" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
</tbody>
</table>

**执行顺序：**时空下采样是 **先空间、后时间**；时空上采样是 **先时间、后空间**。

#### 残差块、缓存与 Attention

- **残差块：**主分支执行两组“RMSNorm → SiLU → `3×3×3` 因果卷积”，再加 shortcut；通道不同时，shortcut 通过 `1×1×1` 卷积对齐。残差块保持 <img src="https://www.zhihu.com/equation?tex=T%2CH%2CW" alt="T,H,W" class="ee_img tr_noresize" eeimg="1">。
- **因果卷积缓存：**普通时间核为 3、步长为 1 的因果卷积，需要前 **2 个时间位置**的特征；序列开头不足处补零。缓存由各层分别维护，时间位置以该层分辨率为准。上表时间下采样使用的是单独的 **1 帧缓存**路径。
- **帧内空间 Attention：**只在同一时间位置的二维特征图上计算，保持 `[C,T,H,W]`；跨时间的信息通过因果卷积传递。

以 Wan2.1-VAE 为基础，我们简单介绍一下其他后续模型。

Wan2.2-VAE 延续了 Wan2.1-VAE 的因果卷积和 <img src="https://www.zhihu.com/equation?tex=4n%2B1" alt="4n+1" class="ee_img tr_noresize" eeimg="1"> 处理方式，时间压缩倍率仍为 4。主要变化是入口新增 `2×2` patchify，将空间位置重排到通道维，再经过三次空间下采样，使空间压缩从 8×8 提升到 16×16，同时将 latent 通道数从 16 增加到 48。结构上，Wan2.2 加宽了编码器和解码器：编码器基础通道数从 96 增至 160，瓶颈通道数从 384 增至 640；解码器基础通道数从 96 增至 256，瓶颈通道数从 384 增至 1024。同时，在原有残差块之外，为整个采样级增加 shortcut：下采样通过重排与 Avgpool 对齐形状，上采样通过重排与 copy对齐形状，再与主分支相加。

HunyuanVideo-1.5 保持 4× 时间压缩，将空间高宽各自的压缩倍率从 8× 提高到 16×，latent channels 从 16 增至 32。相比 Wan 的分离式时空采样，它采用 causal convolution 加 PixelShuffle/Unshuffle 式重排，并在采样模块和 latent 两端的投影中增加 residual shortcuts。中间 Attention 则从帧内的 spatial attention 扩展为 spatiotemporal causal attention，能够直接关注当前及历史帧，实现跨帧信息交互。

## 3. MiniMax-H3 VAE 

接着来看 MiniMax-H3 VAE。MiniMax H3 的默认视频分块规则要求长度对齐到 **<img src="https://www.zhihu.com/equation?tex=17n%2B5" alt="17n+5" class="ee_img tr_noresize" eeimg="1">**（H3 在这方面的处理非常神奇，虽然我想过一些可能的原因，但是还没找到官方依据），如果原视频是 **81 帧（<img src="https://www.zhihu.com/equation?tex=4d%2B1" alt="4d+1" class="ee_img tr_noresize" eeimg="1">）**，本例先**复制末帧补到 90 帧**，重建后裁掉末尾 9 帧。

**整段视频：**`[3,90,480,832]` → **latent：**`[24,27,30,52]` → **重建视频：**`[3,90,480,832]`。

<img src="https://www.zhihu.com/equation?tex=%0A90%3D17%5Ctimes5%2B5%2C%5Cqquad%20T_z%3D5%5Ctimes5%2B2%3D27%2C%5Cqquad%20H_z%3D%5Cfrac%7B480%7D%7B16%7D%3D30%2C%5Cqquad%20W_z%3D%5Cfrac%7B832%7D%7B16%7D%3D52.%0A" alt="&#10;90=17\times5+5,\qquad T_z=5\times5+2=27,\qquad H_z=\frac{480}{16}=30,\qquad W_z=\frac{832}{16}=52.&#10;" class="ee_img tr_noresize" eeimg="1">

### 3.1 Encoder：跟踪一个 17 帧视频块 

编码时，90 帧还会在内部复制末帧，临时补到 **102 帧**，再分成 **6 个 17 帧块**独立编码。

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>当前视频块</td>
<td><code>[3,17,480,832]</code></td>
</tr>
<tr>
<td>入口</td>
<td><code>3×3×3</code> 因果卷积，<strong>3→128</strong></td>
<td><code>[128,17,480,832]</code></td>
</tr>
<tr>
<td>第一级</td>
<td>残差块 ×2，输出 128 通道 → <strong>空间下采样</strong></td>
<td><code>[128,17,240,416]</code></td>
</tr>
<tr>
<td>第二级</td>
<td>残差块 ×2，输出 256 通道 → <strong>时空下采样</strong></td>
<td><code>[256,9,120,208]</code></td>
</tr>
<tr>
<td>第三级</td>
<td>残差块 ×2，输出 256 通道 → <strong>时空下采样</strong></td>
<td><code>[256,5,60,104]</code></td>
</tr>
<tr>
<td>第四级</td>
<td>残差块 ×2，输出 512 通道 → <strong>空间下采样</strong></td>
<td><code>[512,5,30,52]</code></td>
</tr>
<tr>
<td>第五级</td>
<td>残差块 ×2，输出 512 通道</td>
<td><code>[512,5,30,52]</code></td>
</tr>
<tr>
<td>第六级</td>
<td>残差块 ×2，输出 1024 通道</td>
<td><code>[1024,5,30,52]</code></td>
</tr>
<tr>
<td>出口</td>
<td>GroupNorm → SiLU → 因果卷积，<strong>1024→48</strong></td>
<td><code>[48,5,30,52]</code></td>
</tr>
<tr>
<td>分布投影</td>
<td><code>1×1×1</code> 卷积，<strong>48→48</strong></td>
<td><code>[48,5,30,52]</code></td>
</tr>
</tbody>
</table>

两次时间下采样均向上取整，时间长度依次为 **<img src="https://www.zhihu.com/equation?tex=17%5Crightarrow9%5Crightarrow5" alt="17 ->9 ->5" class="ee_img tr_noresize" eeimg="1">**。全部 6 个块编码完成后，沿时间维拼接得到 `[48,30,30,52]`，再按 `token_drop=3` **从整段末尾丢弃 3 个时间位置**，得到 `[48,27,30,52]`（丢弃的部分对应复制的 12 个末帧）。

随后拆分为 <img src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" class="ee_img tr_noresize" eeimg="1"> 和 `log_var`，各为 `[24,27,30,52]`；从后验分布采样得到 latent，再按通道标准化。

### 3.2 Decoder：跟踪一个 latent 窗口 

先对完整 latent **撤销标准化**，再按 **7 个时间位置一窗、每次前进 5 个位置**解码，相邻窗口重叠 2 个位置。下面跟踪其中一个 `[24,7,30,52]` 的窗口；其视频 token 数和隐藏维度分别为：

<img src="https://www.zhihu.com/equation?tex=%0AN%3D7%5Ctimes30%5Ctimes52%3D10920%2C%5Cqquad%20D%3D32%5Ctimes64%3D2048.%0A" alt="&#10;N=7\times30\times52=10920,\qquad D=32\times64=2048.&#10;" class="ee_img tr_noresize" eeimg="1">

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>撤销标准化后的 latent 窗口</td>
<td><code>[24,7,30,52]</code></td>
</tr>
<tr>
<td>入口</td>
<td><code>1×1×1</code> 卷积，<strong>24→24</strong></td>
<td><code>[24,7,30,52]</code></td>
</tr>
<tr>
<td>展平</td>
<td>每个时空位置转为一个 token</td>
<td><code>[10920,24]</code></td>
</tr>
<tr>
<td>特征投影</td>
<td>Linear，<strong>24→2048</strong></td>
<td><code>[10920,2048]</code></td>
</tr>
<tr>
<td>附加 token</td>
<td>追加 <strong>4 个 register token + 1 个初始为零的 token</strong>（牢 CV 操作）</td>
<td><code>[10925,2048]</code></td>
</tr>
<tr>
<td>主体</td>
<td>Transformer block ×36，3D RoPE</td>
<td><code>[10925,2048]</code></td>
</tr>
<tr>
<td>出口</td>
<td>LayerNorm → Linear，<strong>2048→3072</strong></td>
<td><code>[10925,3072]</code></td>
</tr>
<tr>
<td>去除附加 token</td>
<td>保留视频 token</td>
<td><code>[10920,3072]</code></td>
</tr>
<tr>
<td>像素重排</td>
<td>每个 token 展开成 <strong><code>4×16×16</code> RGB 块</strong></td>
<td><code>[3,28,480,832]</code></td>
</tr>
</tbody>
</table>

出口将每个视频 token 投影为 **<img src="https://www.zhihu.com/equation?tex=3072%3D3%5Ctimes4%5Ctimes16%5Ctimes16" alt="3072=3\times4\times16\times16" class="ee_img tr_noresize" eeimg="1">** 个值，再重排成一个 `4×16×16` RGB 块。因此，一个窗口先得到 `[3,28,480,832]`，随后还要做时间裁剪和窗口融合。

#### 时间裁剪与整段拼接 

每个窗口输出的 28 帧先分成前 **20 帧**和后 **8 帧**，两部分各裁掉开头 3 帧，留下 **17 帧主体 + 5 帧重叠区域**。

27 个 latent 时间位置共形成 **5 个窗口**。从 0 开始计数，窗口范围依次为 `0-6`、`5-11`、`10-16`、`15-21`、`20-26`。每窗保留 22 帧，相邻窗口融合重叠的 5 帧，最终得到：

<img src="https://www.zhihu.com/equation?tex=%0A5%5Ctimes22-4%5Ctimes5%3D90%5Ctext%7B%20%E5%B8%A7%7D.%0A" alt="&#10;5\times22-4\times5=90\text{ 帧}.&#10;" class="ee_img tr_noresize" eeimg="1">

融合后的形状为 `[3,90,480,832]`。再裁掉本例额外补入的末尾 9 帧，即恢复为 `[3,81,480,832]`。

### 3.3 下采样与像素展开 

<table>
<thead>
<tr>
<th>操作</th>
<th>具体实现</th>
<th>形状变化</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>空间下采样</strong></td>
<td>右、下各反射补 1 → <code>3×3×3</code> 因果卷积，步长 <code>(1,2,2)</code></td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各减半；<img src="https://www.zhihu.com/equation?tex=C%2CT" alt="C,T" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>时空下采样</strong></td>
<td>右、下各反射补 1 → <code>3×3×3</code> 因果卷积，步长 <code>(2,2,2)</code></td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各减半；<img src="https://www.zhihu.com/equation?tex=T%5Cto%5Clceil%20T%2F2%5Crceil" alt="T\to\lceil T/2\rceil" class="ee_img tr_noresize" eeimg="1">；<img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>像素展开</strong></td>
<td>Linear → 时空重排</td>
<td>每个视频 token 输出一个 <code>4×16×16</code> RGB 块</td>
</tr>
</tbody>
</table>

**下采样顺序：**空间 → 时空 → 时空 → 空间，分别位于前四级末尾。两种下采样中的时间卷积都在左侧补 2 个零位置。

#### 残差块、归一化与 Attention

- **残差块：**主分支执行两组“GroupNorm → SiLU → `3×3×3` 因果卷积”，再加 shortcut；通道不同时，shortcut 通过 `1×1×1` 卷积对齐。残差块保持 <img src="https://www.zhihu.com/equation?tex=T%2CH%2CW" alt="T,H,W" class="ee_img tr_noresize" eeimg="1">。
- **因果编码：**时间卷积只在左侧补零，GroupNorm 按时间位置独立计算。各个 17 帧块独立进入 Encoder。
- **ViT 解码块：**依次执行 RMSNorm → Attention，以及 RMSNorm → 门控 SiLU FFN；两个分支分别经过可学习缩放后加回残差，保持 `[N,2048]`。
- **Attention 范围：**发布配置中的 ViT Decoder 为非因果结构，可以同时关注当前窗口内的各个时间位置。

## 4. LTX-2.5 Video VAE 

接着来看 LTX-2.5 的视频 VAE。它采用**因果 3D CNN Encoder**，提供两种 Decoder：**非因果 3D CNN 解码器**和**非因果 NA Transformer 扩散解码器**。两者使用相同的 latent 空间，时间压缩倍率为 **8**，空间高宽各压缩 **32** 倍，latent 通道数为 **128**。

同样以 **81 帧、480×832** 的 RGB 视频为例，形状统一写作 `[C,T,H,W]`，省略 `B=1`：

**整段视频：**`[3,81,480,832]` → **latent：**`[128,11,15,26]` → **重建视频：**`[3,81,480,832]`。

<img src="https://www.zhihu.com/equation?tex=%0AT_z%3D1%2B%5Cfrac%7B81-1%7D%7B8%7D%3D11%2C%5Cqquad%20H_z%3D%5Cfrac%7B480%7D%7B32%7D%3D15%2C%5Cqquad%20W_z%3D%5Cfrac%7B832%7D%7B32%7D%3D26.%0A" alt="&#10;T_z=1+\frac{81-1}{8}=11,\qquad H_z=\frac{480}{32}=15,\qquad W_z=\frac{832}{32}=26.&#10;" class="ee_img tr_noresize" eeimg="1">

下面跟踪整段视频的 forward 过程。分块与重叠融合属于另外的执行策略，不将视频预先视为 11 个独立编码、解码的小块。

### 4.1 Encoder：跟踪整段 81 帧视频 

入口先做 **4×4 空间 patchify**：将每帧相邻的 16 个像素位置重排到通道维，时间长度不变。随后经过**空间 → 时间 → 时空 → 时空**四级下采样。

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>RGB 视频</td>
<td><code>[3,81,480,832]</code></td>
</tr>
<tr>
<td>Patchify</td>
<td>空间 <code>4×4</code> 重排，<strong>3→48</strong></td>
<td><code>[48,81,120,208]</code></td>
</tr>
<tr>
<td>入口</td>
<td><code>3×3×3</code> 因果卷积，<strong>48→128</strong></td>
<td><code>[128,81,120,208]</code></td>
</tr>
<tr>
<td>第一级</td>
<td>残差块 ×4，128 通道 → <strong>空间下采样，128→256</strong></td>
<td><code>[256,81,60,104]</code></td>
</tr>
<tr>
<td>第二级</td>
<td>残差块 ×6，256 通道 → <strong>时间下采样，256→512</strong></td>
<td><code>[512,41,60,104]</code></td>
</tr>
<tr>
<td>第三级</td>
<td>残差块 ×4，512 通道 → <strong>时空下采样，512→1024</strong></td>
<td><code>[1024,21,30,52]</code></td>
</tr>
<tr>
<td>第四级</td>
<td>残差块 ×2，1024 通道 → <strong>时空下采样，1024→1024</strong></td>
<td><code>[1024,11,15,26]</code></td>
</tr>
<tr>
<td>瓶颈</td>
<td>残差块 ×2，1024 通道</td>
<td><code>[1024,11,15,26]</code></td>
</tr>
<tr>
<td>出口</td>
<td>PixelNorm → SiLU → 因果卷积，<strong>1024→129</strong></td>
<td><code>[129,11,15,26]</code></td>
</tr>
</tbody>
</table>

三次时间下采样均保留首帧对应的位置，时间长度依次为：

<img src="https://www.zhihu.com/equation?tex=%0A81%5Crightarrow41%5Crightarrow21%5Crightarrow11.%0A" alt="&#10;81 ->41 ->21 ->11.&#10;" class="ee_img tr_noresize" eeimg="1">

卷积头输出的前 **128 个通道是 <img src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" class="ee_img tr_noresize" eeimg="1">**。类似地，再使用逐通道均值和标准差归一化，得到 `[128,11,15,26]`。只是这里相比其他的 VAE 没有额外的 `1×1×1` 分布投影卷积。

### 4.2 CNN Decoder：逐级恢复视频 

先撤销 latent 的逐通道归一化，再经过卷积残差块和四级上采样，最后通过 unpatchify 恢复 RGB 像素。

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>撤销归一化后的 latent</td>
<td><code>[128,11,15,26]</code></td>
</tr>
<tr>
<td>入口</td>
<td><code>3×3×3</code> 卷积，<strong>128→1024</strong></td>
<td><code>[1024,11,15,26]</code></td>
</tr>
<tr>
<td>第一级</td>
<td>残差块 ×2，1024 通道 → <strong>时空上采样，1024→512</strong></td>
<td><code>[512,21,30,52]</code></td>
</tr>
<tr>
<td>第二级</td>
<td>残差块 ×2，512 通道 → <strong>时空上采样，512→512</strong></td>
<td><code>[512,41,60,104]</code></td>
</tr>
<tr>
<td>第三级</td>
<td>残差块 ×4，512 通道 → <strong>时间上采样，512→256</strong></td>
<td><code>[256,81,60,104]</code></td>
</tr>
<tr>
<td>第四级</td>
<td>残差块 ×6，256 通道 → <strong>空间上采样，256→128</strong></td>
<td><code>[128,81,120,208]</code></td>
</tr>
<tr>
<td>第五级</td>
<td>残差块 ×4，128 通道</td>
<td><code>[128,81,120,208]</code></td>
</tr>
<tr>
<td>出口</td>
<td>PixelNorm → SiLU → <code>3×3×3</code> 卷积，<strong>128→48</strong></td>
<td><code>[48,81,120,208]</code></td>
</tr>
<tr>
<td>Unpatchify</td>
<td>将通道重排成 <code>4×4</code> 空间像素块，<strong>48→3</strong></td>
<td><code>[3,81,480,832]</code></td>
</tr>
</tbody>
</table>

时间上采样先将长度翻倍，再删除最前面的一个时间位置，因此每次为 **<img src="https://www.zhihu.com/equation?tex=T%5Crightarrow2T-1" alt="T ->2T-1" class="ee_img tr_noresize" eeimg="1">**：

<img src="https://www.zhihu.com/equation?tex=%0A11%5Crightarrow21%5Crightarrow41%5Crightarrow81.%0A" alt="&#10;11 ->21 ->41 ->81.&#10;" class="ee_img tr_noresize" eeimg="1">

该 Decoder 的卷积以**非因果模式**执行，能够利用前后时间位置的信息。

### 4.3 扩散 Decoder：条件特征与像素去噪 

它先将 latent 上采样为条件特征 **context**，再以 context 为条件，对像素噪声进行重建，**1 次去噪、直接预测重建视频 <img src="https://www.zhihu.com/equation?tex=x_0" alt="x_0" class="ee_img tr_noresize" eeimg="1">**。

#### 生成条件特征 

官方实现会将最后一个 latent 时间位置额外复制 **2 次**，用于处理邻域 Attention 的末端边界。我们计入这部分临时补帧，并在最后裁掉。

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>撤销归一化后的 latent</td>
<td><code>[128,11,15,26]</code></td>
</tr>
<tr>
<td>边界处理</td>
<td>将最后一个 latent 时间位置额外复制 2 次</td>
<td><code>[128,13,15,26]</code></td>
</tr>
<tr>
<td>入口</td>
<td>逐位置 Linear，<strong>128→2048</strong></td>
<td><code>[2048,13,15,26]</code></td>
</tr>
<tr>
<td>第一级</td>
<td>NA Block ×4 → <strong>空间上采样，2048→1024</strong></td>
<td><code>[1024,13,30,52]</code></td>
</tr>
<tr>
<td>第二级</td>
<td>NA Block ×6 → <strong>时间上采样，1024→512</strong></td>
<td><code>[512,25,30,52]</code></td>
</tr>
<tr>
<td>第三级</td>
<td>NA Block ×4 → <strong>时空上采样，512→512</strong></td>
<td><code>[512,49,60,104]</code></td>
</tr>
<tr>
<td>第四级</td>
<td>NA Block ×2 → <strong>时空上采样，512→256</strong></td>
<td><code>[256,97,120,208]</code></td>
</tr>
<tr>
<td>裁剪</td>
<td>去掉临时补帧对应的末尾 <strong>16 个时间位置</strong></td>
<td><code>[256,81,120,208]</code></td>
</tr>
</tbody>
</table>

补入的 2 个 latent 时间位置经过 8 倍时间展开，对应 **16 个时间位置**，因此裁剪后为 **<img src="https://www.zhihu.com/equation?tex=97-16%3D81" alt="97-16=81" class="ee_img tr_noresize" eeimg="1">**。得到的 context 已经具有目标视频的时间长度，空间分辨率仍是目标的 **1/4×1/4**。

#### 像素去噪与重建 

初始化与目标视频同形状的随机像素噪声。下面跟踪像素特征分支，context 在各个扩散块中作为条件注入。

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>随机像素噪声 <img src="https://www.zhihu.com/equation?tex=x_t" alt="x_t" class="ee_img tr_noresize" eeimg="1"></td>
<td><code>[3,81,480,832]</code></td>
</tr>
<tr>
<td>Patchify</td>
<td>空间 <code>4×4</code> 重排，<strong>3→48</strong></td>
<td><code>[48,81,120,208]</code></td>
</tr>
<tr>
<td>入口</td>
<td>逐位置 Linear，<strong>48→256</strong></td>
<td><code>[256,81,120,208]</code></td>
</tr>
<tr>
<td>去噪主干</td>
<td>Diffusion NA Block ×8，注入 context 和去噪时间步条件</td>
<td><code>[256,81,120,208]</code></td>
</tr>
<tr>
<td>出口</td>
<td>RMSNorm → Linear，<strong>256→48</strong></td>
<td><code>[48,81,120,208]</code></td>
</tr>
<tr>
<td>Unpatchify</td>
<td>将通道重排成 <code>4×4</code> RGB 像素块，<strong>48→3</strong></td>
<td><code>[3,81,480,832]</code></td>
</tr>
</tbody>
</table>

最后输出重建视频 <img src="https://www.zhihu.com/equation?tex=x_0" alt="x_0" class="ee_img tr_noresize" eeimg="1">。这里的 **8 个 Block 是网络深度**；“去噪时间步”表示噪声等级，与视频的 81 个时间位置是不同概念。

### 4.4 NA Attention：局部时空注意力 

**NA（Neighborhood Attention）就是邻域注意力。**对于特征图上的每个位置 `(t,h,w)`，只取附近一个三维窗口中的 K、V，与当前位置的 Q 计算 Attention。窗口随查询位置滑动，相邻窗口重叠。

例如 **`3×7×7`** 窗口，在非边界处覆盖**前一个、当前、后一个时间位置**，每个时间位置取 **7×7** 个空间格点。因此，每个查询、每个注意力头关注 **<img src="https://www.zhihu.com/equation?tex=3%5Ctimes7%5Ctimes7%3D147" alt="3\times7\times7=147" class="ee_img tr_noresize" eeimg="1">** 个位置，包括自身。

<table>
<thead>
<tr>
<th>阶段</th>
<th>特征通道</th>
<th>Attention 头数</th>
<th>每头维度</th>
<th>窗口：时间 × 高 × 宽</th>
<th>每头关注的位置数</th>
</tr>
</thead>
<tbody>
<tr>
<td>第一级</td>
<td>2048</td>
<td>32</td>
<td>64</td>
<td><code>3×7×7</code></td>
<td>147</td>
</tr>
<tr>
<td>第二级</td>
<td>1024</td>
<td>16</td>
<td>64</td>
<td><code>3×7×7</code></td>
<td>147</td>
</tr>
<tr>
<td>第三级</td>
<td>512</td>
<td>8</td>
<td>64</td>
<td><code>3×5×5</code></td>
<td>75</td>
</tr>
<tr>
<td>第四级</td>
<td>512</td>
<td>8</td>
<td>64</td>
<td><code>3×5×5</code></td>
<td>75</td>
</tr>
<tr>
<td>第五级：像素去噪</td>
<td>256</td>
<td>4</td>
<td>64</td>
<td><code>11×11×11</code></td>
<td>1331</td>
</tr>
</tbody>
</table>

所有阶段的每头维度都是 **64**，头数为 **<img src="https://www.zhihu.com/equation?tex=C%2F64" alt="C/64" class="ee_img tr_noresize" eeimg="1">**。窗口大小按**所在层的特征格点**计算；NA 本身保持 `[C,T,H,W]`，上采样由另外的模块完成。

普通 NA Block 依次执行：

```text
RMSNorm -> 三维 NA Attention -> 加残差
RMSNorm -> SwiGLU MLP        -> 加残差
```

Q、K 使用每头 RMSNorm 和三维 RoPE；MLP 隐藏维度为 **4C**。第五级的 Diffusion NA Block 额外注入 context，并通过去噪时间步调制特征。

这些 Attention 都是**非因果的**，允许关注后续时间位置。到达边界时，窗口向有效区域内平移以保持大小，例如时间窗口为 3 时，首位置关注 `0、1、2`。

### 4.5 上下采样与时间边界 

<table>
<thead>
<tr>
<th>操作</th>
<th>具体实现</th>
<th>形状变化</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>空间 Patchify</strong></td>
<td>将每帧 <code>4×4</code> 像素块重排到通道维</td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各除以 4；<img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 乘以 16；<img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>空间下采样</strong></td>
<td>因果卷积 → 空间重排到通道；加重排、分组平均得到的 shortcut</td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各减半；<img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>时间下采样</strong></td>
<td>复制首帧 1 次 → 因果卷积与时间重排；加 shortcut</td>
<td><img src="https://www.zhihu.com/equation?tex=T%5Crightarrow%28T%2B1%29%2F2" alt="T ->(T+1)/2" class="ee_img tr_noresize" eeimg="1">；<img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>时空下采样</strong></td>
<td>复制首帧 1 次 → 因果卷积与 <code>2×2×2</code> 联合重排；加 shortcut</td>
<td><img src="https://www.zhihu.com/equation?tex=T%5Crightarrow%28T%2B1%29%2F2" alt="T ->(T+1)/2" class="ee_img tr_noresize" eeimg="1">；<img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各减半</td>
</tr>
<tr>
<td><strong>CNN 上采样</strong></td>
<td>卷积生成所需通道 → 通道重排到指定时空维度</td>
<td>指定轴放大 2 倍；时间放大后删除最前面 1 帧</td>
</tr>
<tr>
<td><strong>NA Decoder 上采样</strong></td>
<td>Linear 生成所需通道 → 通道重排到指定时空维度</td>
<td>指定轴放大 2 倍；时间放大后删除最前面 1 帧</td>
</tr>
<tr>
<td><strong>空间 Unpatchify</strong></td>
<td>将通道重排回每帧的 <code>4×4</code> 像素块</td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各乘以 4；<img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 除以 16；<img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
</tbody>
</table>

下采样中的主分支先通过卷积调整通道，再将局部时空位置并入通道维；shortcut 则对输入做同样的重排，并通过分组平均匹配输出通道。时空采样采用联合重排。

#### 残差块、归一化与分块

- **CNN 残差块：**主分支主要执行两组“PixelNorm → SiLU → `3×3×3` 卷积”，再加 shortcut，保持 <img src="https://www.zhihu.com/equation?tex=T%2CH%2CW" alt="T,H,W" class="ee_img tr_noresize" eeimg="1">。
- **PixelNorm：**在每个时空位置上，沿通道维做 RMS 归一化。
- **时间边界：**Encoder 的因果卷积通过复制首帧完成左侧时间填充；含时间下采样的重排模块还会额外复制首帧 1 次。Decoder 则在时间展开后删除前导帧，使整体长度满足 **<img src="https://www.zhihu.com/equation?tex=T_%7B%5Cmathrm%7Bout%7D%7D%3D8%28T_z-1%29%2B1" alt="T_{\mathrm{out}}=8(T_z-1)+1" class="ee_img tr_noresize" eeimg="1">**。
- **分块执行：**LTX 支持带重叠区域的 tiling。这里的 **<img src="https://www.zhihu.com/equation?tex=81%3D1%2B10%5Ctimes8" alt="81=1+10\times8" class="ee_img tr_noresize" eeimg="1">** 表示时间压缩关系；实际解码需要保留跨时间上下文，不能将 11 个 latent 时间位置分别独立解码后直接拼接。

## 5. FLUX 3 Video VAE 

接着来看 FLUX 3 Action 的视频 VAE。它采用**因果 NA Transformer Encoder + 非因果 NA Transformer Decoder**，时间压缩倍率为 **4**，空间高宽各压缩 **32** 倍，latent 通道数为 **96**。

同样以 **81 帧、480×832** 的 RGB 视频为例，形状统一写作 `[C,T,H,W]`，省略 `B=1`：

**整段视频：**`[3,81,480,832]` → **latent：**`[96,21,15,26]` → **重建视频：**`[3,81,480,832]`。

<img src="https://www.zhihu.com/equation?tex=%0AT_z%3D1%2B%5Cfrac%7B81-1%7D%7B4%7D%3D21%2C%5Cqquad%20H_z%3D%5Cfrac%7B480%7D%7B32%7D%3D15%2C%5Cqquad%20W_z%3D%5Cfrac%7B832%7D%7B32%7D%3D26.%0A" alt="&#10;T_z=1+\frac{81-1}{4}=21,\qquad H_z=\frac{480}{32}=15,\qquad W_z=\frac{832}{32}=26.&#10;" class="ee_img tr_noresize" eeimg="1">

### 5.1 Block 与 Attention 

FLUX 3 的 VAE 也使用了 Neighborhood Attention，统一使用 **`5×5×5`** 窗口，还加了 QK Norm 与三维 RoPE；**Encoder 使用时间因果 Attention，Decoder 使用非因果 Attention**。

<table>
<thead>
<tr>
<th>特征通道</th>
<th>Attention 头数</th>
<th>每头维度</th>
<th>窗口：时间 × 高 × 宽</th>
</tr>
</thead>
<tbody>
<tr>
<td>256</td>
<td>4</td>
<td>64</td>
<td><code>5×5×5</code></td>
</tr>
<tr>
<td>512</td>
<td>8</td>
<td>64</td>
<td><code>5×5×5</code></td>
</tr>
<tr>
<td>1024</td>
<td>16</td>
<td>64</td>
<td><code>5×5×5</code></td>
</tr>
<tr>
<td>2048</td>
<td>32</td>
<td>64</td>
<td><code>5×5×5</code></td>
</tr>
</tbody>
</table>

单层 NA 中，Encoder 在时间上只关注**当前位置及最多前 4 个位置**；Decoder 在非边界处关注**前 2 个、当前、后 2 个位置**，边界处将窗口向序列内部平移。单帧输入使用 **`5×5` 二维 NA**。

### 5.2 实际分块与上下文 

**编码分块：**按 **45 帧一块、每次前进 44 帧**执行，相邻块重叠 1 帧，各块独立进入 Encoder。对于经典的 81 帧，会先复制末帧，将 **81 帧补到 89 帧**：

```text
第 1 块：第 1～45 帧
第 2 块：第 45～89 帧
```

每块编码得到 **12 个 latent 时间位置**。保留首块全部输出，后续块丢弃首个 latent 时间位置，再拼接并按原视频长度裁剪：

<img src="https://www.zhihu.com/equation?tex=%0A12%2B%2812-1%29%3D23%0A%5Cquad%5Clongrightarrow%5Cquad%0A%5Ctext%7B%E4%BF%9D%E7%95%99%E5%89%8D%20%7D21%5Ctext%7B%20%E4%B8%AA%E6%97%B6%E9%97%B4%E4%BD%8D%E7%BD%AE%7D.%0A" alt="&#10;12+(12-1)=23&#10;\quad\longrightarrow\quad&#10;\text{保留前 }21\text{ 个时间位置}.&#10;" class="ee_img tr_noresize" eeimg="1">

### 5.3 Encoder：跟踪一个 45 帧视频块 

入口按 **`1×4×4`** 划分像素块并投影到 256 通道，实现为 kernel 和 stride 均为 `1×4×4` 的 Conv3d。随后进行三次空间合并，再在低空间分辨率上完成两次时间合并。

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>当前视频块</td>
<td><code>[3,45,480,832]</code></td>
</tr>
<tr>
<td>入口</td>
<td><code>1×4×4 Conv3d</code>，步长 <code>(1,4,4)</code>，<strong>3→256</strong></td>
<td><code>[256,45,120,208]</code></td>
</tr>
<tr>
<td>第一级</td>
<td>NA Block ×1，256 通道 → <strong>空间合并，256→512</strong></td>
<td><code>[512,45,60,104]</code></td>
</tr>
<tr>
<td>第二级</td>
<td>NA Block ×4，512 通道 → <strong>空间合并，512→1024</strong></td>
<td><code>[1024,45,30,52]</code></td>
</tr>
<tr>
<td>第三级主干</td>
<td>NA Block ×8，1024 通道 → <strong>空间合并，1024→2048</strong></td>
<td><code>[2048,45,15,26]</code></td>
</tr>
<tr>
<td>第一次时间压缩</td>
<td>附加 NA Block ×1 → <strong>时间合并，45→23</strong></td>
<td><code>[2048,23,15,26]</code></td>
</tr>
<tr>
<td>第四级主干</td>
<td>NA Block ×8，2048 通道</td>
<td><code>[2048,23,15,26]</code></td>
</tr>
<tr>
<td>第二次时间压缩</td>
<td>附加 NA Block ×1 → <strong>时间合并，23→12</strong></td>
<td><code>[2048,12,15,26]</code></td>
</tr>
<tr>
<td>出口</td>
<td>逐位置 Linear，<strong>2048→192</strong></td>
<td><code>[192,12,15,26]</code></td>
</tr>
</tbody>
</table>

空间压缩由入口的 4 倍和三次 2 倍空间合并组成：

<img src="https://www.zhihu.com/equation?tex=%0A4%5Ctimes2%5Ctimes2%5Ctimes2%3D32.%0A" alt="&#10;4\times2\times2\times2=32.&#10;" class="ee_img tr_noresize" eeimg="1">

时间长度则依次为：

<img src="https://www.zhihu.com/equation?tex=%0A45%5Crightarrow23%5Crightarrow12.%0A" alt="&#10;45 ->23 ->12.&#10;" class="ee_img tr_noresize" eeimg="1">

出口沿通道拆成两组，各为 `[96,12,15,26]`，只取前一组作为 **<img src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" class="ee_img tr_noresize" eeimg="1">**。随后对 <img src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" class="ee_img tr_noresize" eeimg="1"> 使用模型保存的逐通道均值和标准差归一化。

### 5.4 Decoder：逐级恢复视频 

先撤销 latent 的逐通道归一化，再通过 Linear 投影、NA Block 和时空展开，恢复 RGB 视频。形状统一为 `[C,T,H,W]`，省略 batch 维。

<table>
<thead>
<tr>
<th>顺序</th>
<th>模块与操作</th>
<th>输出形状</th>
</tr>
</thead>
<tbody>
<tr>
<td>输入</td>
<td>撤销归一化后的 latent</td>
<td><code>[96,21,15,26]</code></td>
</tr>
<tr>
<td>入口</td>
<td>逐位置 Linear，<strong>96→2048</strong></td>
<td><code>[2048,21,15,26]</code></td>
</tr>
<tr>
<td>第一级时间展开</td>
<td>NA Block ×8 → <strong>时间展开，21→41</strong> → NA Block ×1</td>
<td><code>[2048,41,15,26]</code></td>
</tr>
<tr>
<td>第一级空间展开</td>
<td><strong>空间展开，2048→1024</strong></td>
<td><code>[1024,41,30,52]</code></td>
</tr>
<tr>
<td>第二级时间展开</td>
<td>NA Block ×8 → <strong>时间展开，41→81</strong> → NA Block ×1</td>
<td><code>[1024,81,30,52]</code></td>
</tr>
<tr>
<td>第二级空间展开</td>
<td><strong>空间展开，1024→512</strong></td>
<td><code>[512,81,60,104]</code></td>
</tr>
<tr>
<td>第三级</td>
<td>NA Block ×4 → <strong>空间展开，512→256</strong></td>
<td><code>[256,81,120,208]</code></td>
</tr>
<tr>
<td>第四级</td>
<td>NA Block ×1，保持 256 通道</td>
<td><code>[256,81,120,208]</code></td>
</tr>
<tr>
<td>出口</td>
<td>逐位置 Linear，<strong>256→48</strong></td>
<td><code>[48,81,120,208]</code></td>
</tr>
<tr>
<td>Unpatchify</td>
<td>将 <strong>48=3×4×4</strong> 个通道重排为 RGB 像素块</td>
<td><code>[3,81,480,832]</code></td>
</tr>
</tbody>
</table>

每次时间展开满足 **<img src="https://www.zhihu.com/equation?tex=T%5Crightarrow2T-1" alt="T ->2T-1" class="ee_img tr_noresize" eeimg="1">**，因此时间长度为：

<img src="https://www.zhihu.com/equation?tex=%0A21%5Crightarrow41%5Crightarrow81.%0A" alt="&#10;21 ->41 ->81.&#10;" class="ee_img tr_noresize" eeimg="1">

### 5.5 上下采样与首帧对齐 

<table>
<thead>
<tr>
<th>操作</th>
<th>具体实现</th>
<th>形状变化</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>入口 Patch Embedding</strong></td>
<td><code>1×4×4 Conv3d</code>，步长 <code>1×4×4</code>，<strong>3→256</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各除以 4；<img src="https://www.zhihu.com/equation?tex=T" alt="T" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>空间合并</strong></td>
<td>拼接 <code>2×2</code> 相邻位置：<code>C→4C</code> → LayerNorm → Linear <strong>4C→2C</strong></td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各减半；<img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 翻倍</td>
</tr>
<tr>
<td><strong>时间合并</strong></td>
<td>奇数长度时在开头复制首位置 → 两位置拼接 → LayerNorm → Linear <strong>2C→C</strong> → 加两位置的均值 shortcut</td>
<td><img src="https://www.zhihu.com/equation?tex=T%5Crightarrow%5Clceil%20T%2F2%5Crceil" alt="T ->\lceil T/2\rceil" class="ee_img tr_noresize" eeimg="1">；<img src="https://www.zhihu.com/equation?tex=C%2CH%2CW" alt="C,H,W" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>空间展开</strong></td>
<td>LayerNorm → Linear <strong>C→4C_out</strong> → 重排成 <code>2×2</code> 空间位置，其中 <code>C_out=C/2</code></td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各翻倍；<img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 减半</td>
</tr>
<tr>
<td><strong>时间展开</strong></td>
<td>LayerNorm → Linear <strong>C→2C</strong> → 加输入在通道维复制两份的 shortcut → 重排到时间维 → 删除首位置</td>
<td><img src="https://www.zhihu.com/equation?tex=T%5Crightarrow2T-1" alt="T ->2T-1" class="ee_img tr_noresize" eeimg="1">；<img src="https://www.zhihu.com/equation?tex=C%2CH%2CW" alt="C,H,W" class="ee_img tr_noresize" eeimg="1"> 不变</td>
</tr>
<tr>
<td><strong>空间 Unpatchify</strong></td>
<td>将每个位置的 48 个通道重排为 <code>3×4×4</code> 像素块</td>
<td><img src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" class="ee_img tr_noresize" eeimg="1"> 各乘以 4；输出 3 通道</td>
</tr>
</tbody>
</table>

**执行顺序：**

- Encoder 第三级：空间合并 → 附加 NA Block → 时间合并；第四级在主干后执行附加 NA Block → 时间合并。
- Decoder 前两级：时间展开 → 附加 NA Block → 空间展开。

**首帧对齐：**编码时通过复制首位置补齐奇数长度，解码时通过删除展开后的首位置恢复对齐。单帧也经过这些模块，但长度仍为 1：

```text
Encoder：1 → 复制成 2 → 合并成 1
Decoder：1 → 展开成 2 → 删除首位置后剩 1
```

两次时间展开后，输出长度为：

<img src="https://www.zhihu.com/equation?tex=%0AT_%7B%5Cmathrm%7Bout%7D%7D%3D2%282T_z-1%29-1%3D4%28T_z-1%29%2B1.%0A" alt="&#10;T_{\mathrm{out}}=2(2T_z-1)-1=4(T_z-1)+1.&#10;" class="ee_img tr_noresize" eeimg="1">
