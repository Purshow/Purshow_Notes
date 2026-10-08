# 视频生成模型 VAE

## 1. 压缩参数与 DiT token 数

倍率按<strong>时间 × 高度 × 宽度</strong>排列，<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C" alt="C" style="vertical-align:middle"> 为 VAE 通道数。末列为 <strong>DiT patchify 后、投影前</strong>的形状，patch 内元素并入通道维。

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=R_%7B%5Cmathrm%7Bnom%7D%7D%3D%5Cfrac%7B3r_t%20r_h%20r_w%7D%7BC%7D%2C%5Cqquad%20R_%7B%5Cmathrm%7Bactual%7D%7D%3D%5Cfrac%7B3THW%7D%7BC%5C%2CT_zH_zW_z%7D%2C%5Cqquad%20N%3D%5Cfrac%7BT_zH_zW_z%7D%7Bp_tp_hp_w%7D." alt="R_{\mathrm{nom}}=\frac{3r_t r_h r_w}{C},\qquad R_{\mathrm{actual}}=\frac{3THW}{C\,T_zH_zW_z},\qquad N=\frac{T_zH_zW_z}{p_tp_hp_w}." style="vertical-align:middle">

DiT patch 为 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=p_t%5Ctimes%20p_h%5Ctimes%20p_w" alt="p_t\times p_h\times p_w" style="vertical-align:middle">，只重排元素；压缩比按 VAE 输出计算，已计入 VAE 内部 patchify。

<table>
<colgroup>
<col style="width: 12%" />
<col style="width: 12%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 12%" />
</colgroup>
<thead>
<tr>
<th>模型</th>
<th>VAE</th>
<th style="text-align: right;">时间 × 高 × 宽压缩</th>
<th style="text-align: right;">Latent<br>通道 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C" alt="C" style="vertical-align:middle"></th>
<th style="text-align: right;">元素压缩比 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=R_%7B%5Cmathrm%7Bnom%7D%7D" alt="R_{\mathrm{nom}}" style="vertical-align:middle"></th>
<th style="text-align: right;">DiT patchify</th>
<th>DiT 输入形状</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>MiniMax H3</strong></td>
<td><strong>H3-VisualVAE</strong><br>因果 3D CNN encoder + <strong>非因果 ViT decoder</strong>；一次投影恢复时空像素块。</td>
<td style="text-align: right;"><strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4%5Ctimes16%5Ctimes16" alt="4\times16\times16" style="vertical-align:middle"></strong></td>
<td style="text-align: right;"><strong>24</strong></td>
<td style="text-align: right;"><strong>128:1</strong></td>
<td style="text-align: right;"><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=1%5Ctimes2%5Ctimes2" alt="1\times2\times2" style="vertical-align:middle"></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C96%2CT_%7B%5Cmathrm%7BH3%7D%7D%2C%5Cfrac%7BH%7D%7B32%7D%2C%5Cfrac%7BW%7D%7B32%7D%5Cright%5D" alt="\left[B,96,T_{\mathrm{H3}},\frac{H}{32},\frac{W}{32}\right]" style="vertical-align:middle"></td>
</tr>
<tr>
<td><strong>LTX-2.5</strong></td>
<td><strong>LTX-2.5 Video VAE</strong><br>因果 3D CNN encoder + <strong>非因果 3D CNN / NA Transformer 扩散 decoder</strong>（CNN 按默认配置）；空间 patchify ×4；扩散版单步像素去噪。</td>
<td style="text-align: right;"><strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=8%5Ctimes32%5Ctimes32" alt="8\times32\times32" style="vertical-align:middle"></strong></td>
<td style="text-align: right;"><strong>128</strong></td>
<td style="text-align: right;"><strong>192:1</strong></td>
<td style="text-align: right;"><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=1%5Ctimes1%5Ctimes1" alt="1\times1\times1" style="vertical-align:middle"></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C128%2C1%2B%5Cfrac%7BT-1%7D%7B8%7D%2C%5Cfrac%7BH%7D%7B32%7D%2C%5Cfrac%7BW%7D%7B32%7D%5Cright%5D" alt="\left[B,128,1+\frac{T-1}{8},\frac{H}{32},\frac{W}{32}\right]" style="vertical-align:middle"></td>
</tr>
<tr>
<td><strong>FLUX 3 Action</strong></td>
<td><strong>FLUX 3 Video VAE</strong><br>因果 NA Transformer encoder + <strong>非因果 NA Transformer decoder</strong>；<code>5×5×5</code> 邻域 attention。</td>
<td style="text-align: right;"><strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4%5Ctimes32%5Ctimes32" alt="4\times32\times32" style="vertical-align:middle"></strong></td>
<td style="text-align: right;"><strong>96</strong></td>
<td style="text-align: right;"><strong>128:1</strong></td>
<td style="text-align: right;"><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=1%5Ctimes1%5Ctimes1" alt="1\times1\times1" style="vertical-align:middle"></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C96%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B32%7D%2C%5Cfrac%7BW%7D%7B32%7D%5Cright%5D" alt="\left[B,96,1+\frac{T-1}{4},\frac{H}{32},\frac{W}{32}\right]" style="vertical-align:middle"></td>
</tr>
<tr>
<td><strong>Wan 2.1 / 2.2 A14B / Lingbot-Video</strong></td>
<td><strong>Wan2.1-VAE</strong><br>因果 3D CNN encoder + 因果 3D CNN decoder；帧内空间 attention、逐层历史缓存。</td>
<td style="text-align: right;"><strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4%5Ctimes8%5Ctimes8" alt="4\times8\times8" style="vertical-align:middle"></strong></td>
<td style="text-align: right;"><strong>16</strong></td>
<td style="text-align: right;"><strong>48:1</strong></td>
<td style="text-align: right;"><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=1%5Ctimes2%5Ctimes2" alt="1\times2\times2" style="vertical-align:middle"></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C64%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B16%7D%2C%5Cfrac%7BW%7D%7B16%7D%5Cright%5D" alt="\left[B,64,1+\frac{T-1}{4},\frac{H}{16},\frac{W}{16}\right]" style="vertical-align:middle"></td>
</tr>
<tr>
<td><strong>Wan 2.2 TI2V-5B</strong></td>
<td><strong>Wan2.2-VAE</strong><br>因果 3D CNN encoder + 因果 3D CNN decoder；空间 patchify ×2、残差采样 shortcut。</td>
<td style="text-align: right;"><strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4%5Ctimes16%5Ctimes16" alt="4\times16\times16" style="vertical-align:middle"></strong></td>
<td style="text-align: right;"><strong>48</strong></td>
<td style="text-align: right;"><strong>64:1</strong></td>
<td style="text-align: right;"><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=1%5Ctimes2%5Ctimes2" alt="1\times2\times2" style="vertical-align:middle"></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C192%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B32%7D%2C%5Cfrac%7BW%7D%7B32%7D%5Cright%5D" alt="\left[B,192,1+\frac{T-1}{4},\frac{H}{32},\frac{W}{32}\right]" style="vertical-align:middle"></td>
</tr>
<tr>
<td><strong>HunyuanVideo-1.5</strong></td>
<td><strong>AutoencoderKLConv3D</strong><br>因果 3D CNN encoder + 因果 3D CNN decoder；瓶颈因果 attention、残差式 3D pixel shuffle。</td>
<td style="text-align: right;"><strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4%5Ctimes16%5Ctimes16" alt="4\times16\times16" style="vertical-align:middle"></strong></td>
<td style="text-align: right;"><strong>32</strong></td>
<td style="text-align: right;"><strong>96:1</strong></td>
<td style="text-align: right;"><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=1%5Ctimes1%5Ctimes1" alt="1\times1\times1" style="vertical-align:middle"></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C32%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B16%7D%2C%5Cfrac%7BW%7D%7B16%7D%5Cright%5D" alt="\left[B,32,1+\frac{T-1}{4},\frac{H}{16},\frac{W}{16}\right]" style="vertical-align:middle"></td>
</tr>
<tr>
<td><strong>MAGI-2 Preview</strong></td>
<td><strong>Wan2.2-VAE + TurboVAED</strong><br>因果 3D CNN encoder + <strong>非因果 3D CNN 蒸馏 decoder</strong>。</td>
<td style="text-align: right;"><strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4%5Ctimes16%5Ctimes16" alt="4\times16\times16" style="vertical-align:middle"></strong></td>
<td style="text-align: right;"><strong>48</strong></td>
<td style="text-align: right;"><strong>64:1</strong></td>
<td style="text-align: right;"><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=1%5Ctimes1%5Ctimes1" alt="1\times1\times1" style="vertical-align:middle"></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cleft%5BB%2C48%2C1%2B%5Cfrac%7BT-1%7D%7B4%7D%2C%5Cfrac%7BH%7D%7B16%7D%2C%5Cfrac%7BW%7D%7B16%7D%5Cright%5D" alt="\left[B,48,1+\frac{T-1}{4},\frac{H}{16},\frac{W}{16}\right]" style="vertical-align:middle"></td>
</tr>
</tbody>
</table>

<strong>H3 的时间压缩比较特殊，严格来说并不是 4:1，元素压缩比也要比 128:1 小一些。</strong>

## 2. Wan2.1-VAE

接着我们具体介绍一下应该是最经典的 VAE：Wan2.1-VAE。以 <strong>81 帧、480×832</strong> 的 RGB 视频为例：

<strong>整段视频：</strong>`[3,81,480,832]` → <strong>latent：</strong>`[16,21,60,104]` → <strong>重建视频：</strong>`[3,81,480,832]`。

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T_z%3D1%2B%5Cfrac%7B81-1%7D%7B4%7D%3D21%2C%5Cqquad%20H_z%3D%5Cfrac%7B480%7D%7B8%7D%3D60%2C%5Cqquad%20W_z%3D%5Cfrac%7B832%7D%7B8%7D%3D104." alt="T_z=1+\frac{81-1}{4}=21,\qquad H_z=\frac{480}{8}=60,\qquad W_z=\frac{832}{8}=104." style="vertical-align:middle">

### 2.1 Encoder：跟踪后续的 4 帧视频块

整段视频按 <strong>1、4、4……帧</strong>分成 <strong>21 个块</strong>：1 个首帧块，加上 20 个后续块，Wan2.1-VAE 会做 Feature Cache，保留前面块的历史特征。下面我们用一个后续块来看看 Wan2.1-VAE 的具体 forward 过程：

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

全部 21 个块编码完成后，先沿时间维拼接，经过一个 `1×1×1` 卷积，再拆分为 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" style="vertical-align:middle"> 和 `log_var`，各为 `[16,21,60,104]`，对 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" style="vertical-align:middle"> 按通道标准化后作为输入。

### 2.2 Decoder：跟踪后续的 latent 时间步

先对完整 latent <strong>撤销标准化</strong>，经过 <strong><code>1×1×1</code> 卷积，16→16</strong>，形状仍为 `[16,21,60,104]`。随后逐个 latent 时间位置解码；下表跟踪一个后续时间步：

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

其中，首帧会跳过时间重采样，只输出 <strong>1 帧</strong>。

### 2.3 上下采样

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各减半；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%2CT" alt="C,T" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>时间下采样</strong></td>
<td>拼接 <strong>1 帧历史缓存</strong> → <code>3×1×1</code> 时间卷积，时间步长 2</td>
<td>后续块 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T" alt="T" style="vertical-align:middle"> 减半；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%2CH%2CW" alt="C,H,W" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>空间上采样</strong></td>
<td>逐帧最近邻插值 ×2（<code>nearest-exact</code>）→ <code>3×3 Conv2d</code>，<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%5Crightarrow%20C%2F2" alt="C\rightarrow C/2" style="vertical-align:middle"></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各翻倍，<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C" alt="C" style="vertical-align:middle"> 减半；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T" alt="T" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>时间上采样</strong></td>
<td><code>3×1×1</code> 因果卷积，<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%5Crightarrow2C" alt="C\rightarrow2C" style="vertical-align:middle"> → 将两组通道重排到时间维</td>
<td>后续块 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T" alt="T" style="vertical-align:middle"> 翻倍；最终 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%2CH%2CW" alt="C,H,W" style="vertical-align:middle"> 不变</td>
</tr>
</tbody>
</table>

<strong>执行顺序：</strong>时空下采样是 <strong>先空间、后时间</strong>；时空上采样是 <strong>先时间、后空间</strong>。

#### 残差块、缓存与 Attention

- <strong>残差块：</strong>主分支执行两组“RMSNorm → SiLU → `3×3×3` 因果卷积”，再加 shortcut；通道不同时，shortcut 通过 `1×1×1` 卷积对齐。残差块保持 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%2CH%2CW" alt="T,H,W" style="vertical-align:middle">。
- <strong>因果卷积缓存：</strong>普通时间核为 3、步长为 1 的因果卷积，需要前 <strong>2 个时间位置</strong>的特征；序列开头不足处补零。缓存由各层分别维护，时间位置以该层分辨率为准。上表时间下采样使用的是单独的 <strong>1 帧缓存</strong>路径。
- <strong>帧内空间 Attention：</strong>只在同一时间位置的二维特征图上计算，保持 `[C,T,H,W]`；跨时间的信息通过因果卷积传递。

以 Wan2.1-VAE 为基础，我们简单介绍一下其他后续模型。

Wan2.2-VAE 延续了 Wan2.1-VAE 的因果卷积和 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4n%2B1" alt="4n+1" style="vertical-align:middle"> 处理方式，时间压缩倍率仍为 4。主要变化是入口新增 `2×2` patchify，将空间位置重排到通道维，再经过三次空间下采样，使空间压缩从 8×8 提升到 16×16，同时将 latent 通道数从 16 增加到 48。结构上，Wan2.2 加宽了编码器和解码器：编码器基础通道数从 96 增至 160，瓶颈通道数从 384 增至 640；解码器基础通道数从 96 增至 256，瓶颈通道数从 384 增至 1024。同时，在原有残差块之外，为整个采样级增加 shortcut：下采样通过重排与 Avgpool 对齐形状，上采样通过重排与 copy对齐形状，再与主分支相加。

HunyuanVideo-1.5 保持 4× 时间压缩，将空间高宽各自的压缩倍率从 8× 提高到 16×，latent channels 从 16 增至 32。相比 Wan 的分离式时空采样，它采用 causal convolution 加 PixelShuffle/Unshuffle 式重排，并在采样模块和 latent 两端的投影中增加 residual shortcuts。中间 Attention 则从帧内的 spatial attention 扩展为 spatiotemporal causal attention，能够直接关注当前及历史帧，实现跨帧信息交互。

## 3. MiniMax-H3 VAE

接着来看 MiniMax-H3 VAE。MiniMax H3 的默认视频分块规则要求长度对齐到 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=17n%2B5" alt="17n+5" style="vertical-align:middle"></strong>（H3 在这方面的处理非常神奇，虽然我想过一些可能的原因，但是还没找到官方依据），如果原视频是 <strong>81 帧（<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4d%2B1" alt="4d+1" style="vertical-align:middle">）</strong>，本例先<strong>复制末帧补到 90 帧</strong>，重建后裁掉末尾 9 帧。

<strong>整段视频：</strong>`[3,90,480,832]` → <strong>latent：</strong>`[24,27,30,52]` → <strong>重建视频：</strong>`[3,90,480,832]`。

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=90%3D17%5Ctimes5%2B5%2C%5Cqquad%20T_z%3D5%5Ctimes5%2B2%3D27%2C%5Cqquad%20H_z%3D%5Cfrac%7B480%7D%7B16%7D%3D30%2C%5Cqquad%20W_z%3D%5Cfrac%7B832%7D%7B16%7D%3D52." alt="90=17\times5+5,\qquad T_z=5\times5+2=27,\qquad H_z=\frac{480}{16}=30,\qquad W_z=\frac{832}{16}=52." style="vertical-align:middle">

### 3.1 Encoder：跟踪一个 17 帧视频块

编码时，90 帧还会在内部复制末帧，临时补到 <strong>102 帧</strong>，再分成 <strong>6 个 17 帧块</strong>独立编码。

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

两次时间下采样均向上取整，时间长度依次为 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=17%5Crightarrow9%5Crightarrow5" alt="17\rightarrow9\rightarrow5" style="vertical-align:middle"></strong>。全部 6 个块编码完成后，沿时间维拼接得到 `[48,30,30,52]`，再按 `token_drop=3` <strong>从整段末尾丢弃 3 个时间位置</strong>，得到 `[48,27,30,52]`（丢弃的部分对应复制的 12 个末帧）。

随后拆分为 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" style="vertical-align:middle"> 和 `log_var`，各为 `[24,27,30,52]`；从后验分布采样得到 latent，再按通道标准化。

### 3.2 Decoder：跟踪一个 latent 窗口

先对完整 latent <strong>撤销标准化</strong>，再按 <strong>7 个时间位置一窗、每次前进 5 个位置</strong>解码，相邻窗口重叠 2 个位置。下面跟踪其中一个 `[24,7,30,52]` 的窗口；其视频 token 数和隐藏维度分别为：

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=N%3D7%5Ctimes30%5Ctimes52%3D10920%2C%5Cqquad%20D%3D32%5Ctimes64%3D2048." alt="N=7\times30\times52=10920,\qquad D=32\times64=2048." style="vertical-align:middle">

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

出口将每个视频 token 投影为 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=3072%3D3%5Ctimes4%5Ctimes16%5Ctimes16" alt="3072=3\times4\times16\times16" style="vertical-align:middle"></strong> 个值，再重排成一个 `4×16×16` RGB 块。因此，一个窗口先得到 `[3,28,480,832]`，随后还要做时间裁剪和窗口融合。

#### 时间裁剪与整段拼接

每个窗口输出的 28 帧先分成前 <strong>20 帧</strong>和后 <strong>8 帧</strong>，两部分各裁掉开头 3 帧，留下 <strong>17 帧主体 + 5 帧重叠区域</strong>。

27 个 latent 时间位置共形成 <strong>5 个窗口</strong>。从 0 开始计数，窗口范围依次为 `0-6`、`5-11`、`10-16`、`15-21`、`20-26`。每窗保留 22 帧，相邻窗口融合重叠的 5 帧，最终得到：

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=5%5Ctimes22-4%5Ctimes5%3D90%5Ctext%7B%20%E5%B8%A7%7D." alt="5\times22-4\times5=90\text{ 帧}." style="vertical-align:middle">

融合后的形状为 `[3,90,480,832]`。再裁掉本例额外补入的末尾 9 帧，即恢复为 `[3,81,480,832]`。

### 3.3 下采样与像素展开

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各减半；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%2CT" alt="C,T" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>时空下采样</strong></td>
<td>右、下各反射补 1 → <code>3×3×3</code> 因果卷积，步长 <code>(2,2,2)</code></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各减半；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%5Cto%5Clceil%20T%2F2%5Crceil" alt="T\to\lceil T/2\rceil" style="vertical-align:middle">；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C" alt="C" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>像素展开</strong></td>
<td>Linear → 时空重排</td>
<td>每个视频 token 输出一个 <code>4×16×16</code> RGB 块</td>
</tr>
</tbody>
</table>

<strong>下采样顺序：</strong>空间 → 时空 → 时空 → 空间，分别位于前四级末尾。两种下采样中的时间卷积都在左侧补 2 个零位置。

#### 残差块、归一化与 Attention

- <strong>残差块：</strong>主分支执行两组“GroupNorm → SiLU → `3×3×3` 因果卷积”，再加 shortcut；通道不同时，shortcut 通过 `1×1×1` 卷积对齐。残差块保持 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%2CH%2CW" alt="T,H,W" style="vertical-align:middle">。
- <strong>因果编码：</strong>时间卷积只在左侧补零，GroupNorm 按时间位置独立计算。各个 17 帧块独立进入 Encoder。
- <strong>ViT 解码块：</strong>依次执行 RMSNorm → Attention，以及 RMSNorm → 门控 SiLU FFN；两个分支分别经过可学习缩放后加回残差，保持 `[N,2048]`。
- <strong>Attention 范围：</strong>发布配置中的 ViT Decoder 为非因果结构，可以同时关注当前窗口内的各个时间位置。

## 4. LTX-2.5 Video VAE

接着来看 LTX-2.5 的视频 VAE。它采用<strong>因果 3D CNN Encoder</strong>，提供两种 Decoder：<strong>非因果 3D CNN 解码器</strong>和<strong>非因果 NA Transformer 扩散解码器</strong>。两者使用相同的 latent 空间，时间压缩倍率为 <strong>8</strong>，空间高宽各压缩 <strong>32</strong> 倍，latent 通道数为 <strong>128</strong>。

同样以 <strong>81 帧、480×832</strong> 的 RGB 视频为例，形状统一写作 `[C,T,H,W]`，省略 `B=1`：

<strong>整段视频：</strong>`[3,81,480,832]` → <strong>latent：</strong>`[128,11,15,26]` → <strong>重建视频：</strong>`[3,81,480,832]`。

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T_z%3D1%2B%5Cfrac%7B81-1%7D%7B8%7D%3D11%2C%5Cqquad%20H_z%3D%5Cfrac%7B480%7D%7B32%7D%3D15%2C%5Cqquad%20W_z%3D%5Cfrac%7B832%7D%7B32%7D%3D26." alt="T_z=1+\frac{81-1}{8}=11,\qquad H_z=\frac{480}{32}=15,\qquad W_z=\frac{832}{32}=26." style="vertical-align:middle">

下面跟踪整段视频的 forward 过程。分块与重叠融合属于另外的执行策略，不将视频预先视为 11 个独立编码、解码的小块。

### 4.1 Encoder：跟踪整段 81 帧视频

入口先做 <strong>4×4 空间 patchify</strong>：将每帧相邻的 16 个像素位置重排到通道维，时间长度不变。随后经过<strong>空间 → 时间 → 时空 → 时空</strong>四级下采样。

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=81%5Crightarrow41%5Crightarrow21%5Crightarrow11." alt="81\rightarrow41\rightarrow21\rightarrow11." style="vertical-align:middle">

卷积头输出的前 <strong>128 个通道是 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" style="vertical-align:middle"></strong>。类似地，再使用逐通道均值和标准差归一化，得到 `[128,11,15,26]`。只是这里相比其他的 VAE 没有额外的 `1×1×1` 分布投影卷积。

### 4.2 CNN Decoder：逐级恢复视频

先撤销 latent 的逐通道归一化，再经过卷积残差块和四级上采样，最后通过 unpatchify 恢复 RGB 像素。

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

时间上采样先将长度翻倍，再删除最前面的一个时间位置，因此每次为 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%5Crightarrow2T-1" alt="T\rightarrow2T-1" style="vertical-align:middle"></strong>：

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=11%5Crightarrow21%5Crightarrow41%5Crightarrow81." alt="11\rightarrow21\rightarrow41\rightarrow81." style="vertical-align:middle">

该 Decoder 的卷积以<strong>非因果模式</strong>执行，能够利用前后时间位置的信息。

### 4.3 扩散 Decoder：条件特征与像素去噪

它先将 latent 上采样为条件特征 <strong>context</strong>，再以 context 为条件，对像素噪声进行重建，<strong>1 次去噪、直接预测重建视频 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=x_0" alt="x_0" style="vertical-align:middle"></strong>。

#### 生成条件特征

官方实现会将最后一个 latent 时间位置额外复制 <strong>2 次</strong>，用于处理邻域 Attention 的末端边界。我们计入这部分临时补帧，并在最后裁掉。

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

补入的 2 个 latent 时间位置经过 8 倍时间展开，对应 <strong>16 个时间位置</strong>，因此裁剪后为 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=97-16%3D81" alt="97-16=81" style="vertical-align:middle"></strong>。得到的 context 已经具有目标视频的时间长度，空间分辨率仍是目标的 <strong>1/4×1/4</strong>。

#### 像素去噪与重建

初始化与目标视频同形状的随机像素噪声。下面跟踪像素特征分支，context 在各个扩散块中作为条件注入。

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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
<td>随机像素噪声 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=x_t" alt="x_t" style="vertical-align:middle"></td>
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

最后输出重建视频 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=x_0" alt="x_0" style="vertical-align:middle">。这里的 <strong>8 个 Block 是网络深度</strong>；“去噪时间步”表示噪声等级，与视频的 81 个时间位置是不同概念。

### 4.4 NA Attention：局部时空注意力

<strong>NA（Neighborhood Attention）就是邻域注意力。</strong>对于特征图上的每个位置 `(t,h,w)`，只取附近一个三维窗口中的 K、V，与当前位置的 Q 计算 Attention。窗口随查询位置滑动，相邻窗口重叠。

例如 <strong><code>3×7×7</code></strong> 窗口，在非边界处覆盖<strong>前一个、当前、后一个时间位置</strong>，每个时间位置取 <strong>7×7</strong> 个空间格点。因此，每个查询、每个注意力头关注 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=3%5Ctimes7%5Ctimes7%3D147" alt="3\times7\times7=147" style="vertical-align:middle"></strong> 个位置，包括自身。

<table>
<colgroup>
<col style="width: 13%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 18%" />
<col style="width: 13%" />
<col style="width: 18%" />
</colgroup>
<thead>
<tr>
<th>阶段</th>
<th style="text-align: right;">特征通道</th>
<th style="text-align: right;">Attention 头数</th>
<th style="text-align: right;">每头维度</th>
<th>窗口：时间 × 高 × 宽</th>
<th style="text-align: right;">每头关注的位置数</th>
</tr>
</thead>
<tbody>
<tr>
<td>第一级</td>
<td style="text-align: right;">2048</td>
<td style="text-align: right;">32</td>
<td style="text-align: right;">64</td>
<td><code>3×7×7</code></td>
<td style="text-align: right;">147</td>
</tr>
<tr>
<td>第二级</td>
<td style="text-align: right;">1024</td>
<td style="text-align: right;">16</td>
<td style="text-align: right;">64</td>
<td><code>3×7×7</code></td>
<td style="text-align: right;">147</td>
</tr>
<tr>
<td>第三级</td>
<td style="text-align: right;">512</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">64</td>
<td><code>3×5×5</code></td>
<td style="text-align: right;">75</td>
</tr>
<tr>
<td>第四级</td>
<td style="text-align: right;">512</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">64</td>
<td><code>3×5×5</code></td>
<td style="text-align: right;">75</td>
</tr>
<tr>
<td>第五级：像素去噪</td>
<td style="text-align: right;">256</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;">64</td>
<td><code>11×11×11</code></td>
<td style="text-align: right;">1331</td>
</tr>
</tbody>
</table>

所有阶段的每头维度都是 <strong>64</strong>，头数为 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%2F64" alt="C/64" style="vertical-align:middle"></strong>。窗口大小按<strong>所在层的特征格点</strong>计算；NA 本身保持 `[C,T,H,W]`，上采样由另外的模块完成。

普通 NA Block 依次执行：

```
RMSNorm -> 三维 NA Attention -> 加残差
RMSNorm -> SwiGLU MLP        -> 加残差
```

Q、K 使用每头 RMSNorm 和三维 RoPE；MLP 隐藏维度为 <strong>4C</strong>。第五级的 Diffusion NA Block 额外注入 context，并通过去噪时间步调制特征。

这些 Attention 都是<strong>非因果的</strong>，允许关注后续时间位置。到达边界时，窗口向有效区域内平移以保持大小，例如时间窗口为 3 时，首位置关注 `0、1、2`。

### 4.5 上下采样与时间边界

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各除以 4；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C" alt="C" style="vertical-align:middle"> 乘以 16；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T" alt="T" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>空间下采样</strong></td>
<td>因果卷积 → 空间重排到通道；加重排、分组平均得到的 shortcut</td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各减半；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T" alt="T" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>时间下采样</strong></td>
<td>复制首帧 1 次 → 因果卷积与时间重排；加 shortcut</td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%5Crightarrow%28T%2B1%29%2F2" alt="T\rightarrow(T+1)/2" style="vertical-align:middle">；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>时空下采样</strong></td>
<td>复制首帧 1 次 → 因果卷积与 <code>2×2×2</code> 联合重排；加 shortcut</td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%5Crightarrow%28T%2B1%29%2F2" alt="T\rightarrow(T+1)/2" style="vertical-align:middle">；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各减半</td>
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
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各乘以 4；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C" alt="C" style="vertical-align:middle"> 除以 16；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T" alt="T" style="vertical-align:middle"> 不变</td>
</tr>
</tbody>
</table>

下采样中的主分支先通过卷积调整通道，再将局部时空位置并入通道维；shortcut 则对输入做同样的重排，并通过分组平均匹配输出通道。时空采样采用联合重排。

#### 残差块、归一化与分块

- <strong>CNN 残差块：</strong>主分支主要执行两组“PixelNorm → SiLU → `3×3×3` 卷积”，再加 shortcut，保持 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%2CH%2CW" alt="T,H,W" style="vertical-align:middle">。
- <strong>PixelNorm：</strong>在每个时空位置上，沿通道维做 RMS 归一化。
- <strong>时间边界：</strong>Encoder 的因果卷积通过复制首帧完成左侧时间填充；含时间下采样的重排模块还会额外复制首帧 1 次。Decoder 则在时间展开后删除前导帧，使整体长度满足 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T_%7B%5Cmathrm%7Bout%7D%7D%3D8%28T_z-1%29%2B1" alt="T_{\mathrm{out}}=8(T_z-1)+1" style="vertical-align:middle"></strong>。
- <strong>分块执行：</strong>LTX 支持带重叠区域的 tiling。这里的 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=81%3D1%2B10%5Ctimes8" alt="81=1+10\times8" style="vertical-align:middle"></strong> 表示时间压缩关系；实际解码需要保留跨时间上下文，不能将 11 个 latent 时间位置分别独立解码后直接拼接。

## 5. FLUX 3 Video VAE

接着来看 FLUX 3 Action 的视频 VAE。它采用<strong>因果 NA Transformer Encoder + 非因果 NA Transformer Decoder</strong>，时间压缩倍率为 <strong>4</strong>，空间高宽各压缩 <strong>32</strong> 倍，latent 通道数为 <strong>96</strong>。

同样以 <strong>81 帧、480×832</strong> 的 RGB 视频为例，形状统一写作 `[C,T,H,W]`，省略 `B=1`：

<strong>整段视频：</strong>`[3,81,480,832]` → <strong>latent：</strong>`[96,21,15,26]` → <strong>重建视频：</strong>`[3,81,480,832]`。

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T_z%3D1%2B%5Cfrac%7B81-1%7D%7B4%7D%3D21%2C%5Cqquad%20H_z%3D%5Cfrac%7B480%7D%7B32%7D%3D15%2C%5Cqquad%20W_z%3D%5Cfrac%7B832%7D%7B32%7D%3D26." alt="T_z=1+\frac{81-1}{4}=21,\qquad H_z=\frac{480}{32}=15,\qquad W_z=\frac{832}{32}=26." style="vertical-align:middle">

### 5.1 Block 与 Attention

FLUX 3 的 VAE 也使用了 Neighborhood Attention，统一使用 <strong><code>5×5×5</code></strong> 窗口，还加了 QK Norm 与三维 RoPE；<strong>Encoder 使用时间因果 Attention，Decoder 使用非因果 Attention</strong>。

<table>
<thead>
<tr>
<th style="text-align: right;">特征通道</th>
<th style="text-align: right;">Attention 头数</th>
<th style="text-align: right;">每头维度</th>
<th>窗口：时间 × 高 × 宽</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: right;">256</td>
<td style="text-align: right;">4</td>
<td style="text-align: right;">64</td>
<td><code>5×5×5</code></td>
</tr>
<tr>
<td style="text-align: right;">512</td>
<td style="text-align: right;">8</td>
<td style="text-align: right;">64</td>
<td><code>5×5×5</code></td>
</tr>
<tr>
<td style="text-align: right;">1024</td>
<td style="text-align: right;">16</td>
<td style="text-align: right;">64</td>
<td><code>5×5×5</code></td>
</tr>
<tr>
<td style="text-align: right;">2048</td>
<td style="text-align: right;">32</td>
<td style="text-align: right;">64</td>
<td><code>5×5×5</code></td>
</tr>
</tbody>
</table>

单层 NA 中，Encoder 在时间上只关注<strong>当前位置及最多前 4 个位置</strong>；Decoder 在非边界处关注<strong>前 2 个、当前、后 2 个位置</strong>，边界处将窗口向序列内部平移。单帧输入使用 <strong><code>5×5</code> 二维 NA</strong>。

### 5.2 实际分块与上下文

<strong>编码分块：</strong>按 <strong>45 帧一块、每次前进 44 帧</strong>执行，相邻块重叠 1 帧，各块独立进入 Encoder。对于经典的 81 帧，会先复制末帧，将 <strong>81 帧补到 89 帧</strong>：

```
第 1 块：第 1～45 帧
第 2 块：第 45～89 帧
```

每块编码得到 <strong>12 个 latent 时间位置</strong>。保留首块全部输出，后续块丢弃首个 latent 时间位置，再拼接并按原视频长度裁剪：

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=12%2B%2812-1%29%3D23%20%5Cquad%5Clongrightarrow%5Cquad%20%5Ctext%7B%E4%BF%9D%E7%95%99%E5%89%8D%20%7D21%5Ctext%7B%20%E4%B8%AA%E6%97%B6%E9%97%B4%E4%BD%8D%E7%BD%AE%7D." alt="12+(12-1)=23 \quad\longrightarrow\quad \text{保留前 }21\text{ 个时间位置}." style="vertical-align:middle">

### 5.3 Encoder：跟踪一个 45 帧视频块

入口按 <strong><code>1×4×4</code></strong> 划分像素块并投影到 256 通道，实现为 kernel 和 stride 均为 `1×4×4` 的 Conv3d。随后进行三次空间合并，再在低空间分辨率上完成两次时间合并。

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=4%5Ctimes2%5Ctimes2%5Ctimes2%3D32." alt="4\times2\times2\times2=32." style="vertical-align:middle">

时间长度则依次为：

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=45%5Crightarrow23%5Crightarrow12." alt="45\rightarrow23\rightarrow12." style="vertical-align:middle">

出口沿通道拆成两组，各为 `[96,12,15,26]`，只取前一组作为 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" style="vertical-align:middle"></strong>。随后对 <img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=%5Cmu" alt="\mu" style="vertical-align:middle"> 使用模型保存的逐通道均值和标准差归一化。

### 5.4 Decoder：逐级恢复视频

先撤销 latent 的逐通道归一化，再通过 Linear 投影、NA Block 和时空展开，恢复 RGB 视频。形状统一为 `[C,T,H,W]`，省略 batch 维。

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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

每次时间展开满足 <strong><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%5Crightarrow2T-1" alt="T\rightarrow2T-1" style="vertical-align:middle"></strong>，因此时间长度为：

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=21%5Crightarrow41%5Crightarrow81." alt="21\rightarrow41\rightarrow81." style="vertical-align:middle">

### 5.5 上下采样与首帧对齐

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
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
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各除以 4；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T" alt="T" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>空间合并</strong></td>
<td>拼接 <code>2×2</code> 相邻位置：<code>C→4C</code> → LayerNorm → Linear <strong>4C→2C</strong></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各减半；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C" alt="C" style="vertical-align:middle"> 翻倍</td>
</tr>
<tr>
<td><strong>时间合并</strong></td>
<td>奇数长度时在开头复制首位置 → 两位置拼接 → LayerNorm → Linear <strong>2C→C</strong> → 加两位置的均值 shortcut</td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%5Crightarrow%5Clceil%20T%2F2%5Crceil" alt="T\rightarrow\lceil T/2\rceil" style="vertical-align:middle">；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%2CH%2CW" alt="C,H,W" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>空间展开</strong></td>
<td>LayerNorm → Linear <strong>C→4C_out</strong> → 重排成 <code>2×2</code> 空间位置，其中 <code>C_out=C/2</code></td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各翻倍；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C" alt="C" style="vertical-align:middle"> 减半</td>
</tr>
<tr>
<td><strong>时间展开</strong></td>
<td>LayerNorm → Linear <strong>C→2C</strong> → 加输入在通道维复制两份的 shortcut → 重排到时间维 → 删除首位置</td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T%5Crightarrow2T-1" alt="T\rightarrow2T-1" style="vertical-align:middle">；<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=C%2CH%2CW" alt="C,H,W" style="vertical-align:middle"> 不变</td>
</tr>
<tr>
<td><strong>空间 Unpatchify</strong></td>
<td>将每个位置的 48 个通道重排为 <code>3×4×4</code> 像素块</td>
<td><img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=H%2CW" alt="H,W" style="vertical-align:middle"> 各乘以 4；输出 3 通道</td>
</tr>
</tbody>
</table>

<strong>执行顺序：</strong>

- Encoder 第三级：空间合并 → 附加 NA Block → 时间合并；第四级在主干后执行附加 NA Block → 时间合并。
- Decoder 前两级：时间展开 → 附加 NA Block → 空间展开。

<strong>首帧对齐：</strong>编码时通过复制首位置补齐奇数长度，解码时通过删除展开后的首位置恢复对齐。单帧也经过这些模块，但长度仍为 1：

```
Encoder：1 → 复制成 2 → 合并成 1
Decoder：1 → 展开成 2 → 删除首位置后剩 1
```

两次时间展开后，输出长度为：

<img class="ee_img tr_noresize" eeimg="1" src="https://www.zhihu.com/equation?tex=T_%7B%5Cmathrm%7Bout%7D%7D%3D2%282T_z-1%29-1%3D4%28T_z-1%29%2B1." alt="T_{\mathrm{out}}=2(2T_z-1)-1=4(T_z-1)+1." style="vertical-align:middle">
