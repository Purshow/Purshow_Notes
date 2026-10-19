# Vision Encoder Architectures and Multimodal Training

**Compression convention:** Overall factors are written as **time × height × width**, measuring the nominal reduction from the input pixel/time grid to the output visual-token grid, including patchification. $N$ denotes the effective token count after patchification and before the final merge. Frame sampling and resizing are excluded from network compression; boundary padding is handled by rounding up. Replicating a static image along the time dimension does not count as actual frame compression; the temporal factor of 1 for images is a placeholder.

**Dynamic-resolution convention:** A variable two-dimensional patch grid and effective token count, allowing resizing/padding within a budget.

| Model | Visual module architecture | Spatiotemporal compression, position encoding, and dynamic resolution | Multimodal training recipe | Visual-input infrastructure optimizations |
| --- | --- | --- | --- | --- |
| **MAI-Base-2** | **The biggest, newest, strongest Encoder-Free model!!!** | — | — | — |
| **Kimi K3** | **MoonViT-V2, 401M, 27 layers, width 1024.**<br>**Attention: MHA, Q/K/V = 12/12/12 heads, head dimension 128.** According to the [official public code](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L603-L615), all $T H W$ tokens within each chunk use non-causal global attention, jointly over space and time.<br>FFN: standard two-layer **GELU-tanh MLP, 1024→4096→1024**; final encoder RMSNorm.<br>Projector: after merging, **4096→4096→7168, GELU, final RMSNorm**. | **Overall: $t×28×28$; $4×28×28$ for a full four-frame chunk.**<br>**Patch / compression:** Conv2d, $1×14×14$. After 27 ViT layers and final RMSNorm, `mean(dim=0)` averages features at the same spatial location and channel across $t$ frames; spatial $2×2$ neighborhoods are then concatenated along channels, **$N→N/(4t)$**. Each chunk has $1≤t≤4$; one forward pass can pack multiple chunks, pooled independently. Separate images use $t=1$.<br>**Position:** bilinearly interpolated learned 64×64 spatial table + fixed temporal encoding; 2D RoPE inside attention, with spatial coordinates repeated across frames.<br>**Dynamic resolution: yes.** Variable $T,H,W$ grids, packed variable-length patches, and chunk isolation via `cu_seqlens`. Up to 3584×3584. | **MoonViT-V2 is trained from scratch, jointly on images and text from the start of main-model training, with an autoregressive objective.**<br>Visual data includes captions, interleaved image-text, OCR, perception, video, visual code, and rendered SVGs, webpages, games, and CAD data. Optimization uses per-head Muon, **1% warmup, cosine scheduling, and weight decay 0.1**.<br>Related post-training: **SFT → domain-expert RL including visual tasks → multi-teacher on-policy distillation**. | Visual context parallelism shards image patches, gathers KV, and dynamically partitions subgroups to balance large-image workloads. Visual-encoder forward and backward passes fill idle intervals in the main training pipeline, reducing waits caused by visual processing. |
| **Qwen3.8-Flash-Next** | **ViT, 415.9M, 27 layers, width 1152.**<br>**Attention: MHA, Q/K/V = 16/16/16 heads, head dimension 72**; non-causal global attention within each temporal slice.<br>FFN: standard two-layer **GELU-tanh MLP, 1152→4304→1152**.<br>Projector: LayerNorm followed by spatial $2×2$ concatenation, **4608→4608→2560, GELU**. | **Overall: $2×32×32$; 2× temporal compression at patchification, with no subsequent temporal reduction.**<br>**Patch / compression:** Conv3d kernel=stride=$2×16×16$; final LayerNorm followed by spatial $2×2$ concatenation + MLP, **$N→N/4$**. Only additional spatial compression; no final temporal pooling.<br>**Position:** bilinearly interpolated 48×48 absolute-position table; 2D RoPE inside visual attention.<br>**Dynamic resolution: yes.** Variable $T,H,W$ grids, packed variable-length patches, and chunk isolation via `cu_seqlens`. | — | — |
| **DeepSeek-V4.1-Flash** | **DeepSeek-ViT, 411.8M, 32 layers, width 1024.**<br>**Attention: MHA, Q/K/V = 16/16/16 heads, head dimension 64**; non-causal global attention within each image.<br>FFN: **SwiGLU, 1024→2816→1024**, three projections; pre-RMSNorm and final encoder RMSNorm.<br>Projector: after spatial unfolding, **9216→5120→5120, GELU**; separate **73.4M** aligner. | **Overall: $1×42×42$.**<br>**Patch / compression:** flatten $1×14×14$ RGB patches + Linear, 588→1024; final `unfold(kernel=stride=3)` performs **$3×3$ pixel-unshuffle, $N→N/9$**. Pad grids that are not multiples of 3; output spatial grid is $⌈h/3⌉×⌈w/3⌉$, using rounded-up dimensions. No temporal pooling in the image forward pass.<br>**Position:** no absolute-position table; visual 2D RoPE, θ=10000.<br>**Dynamic resolution: yes.** Plan the grid from aspect ratio and total token budget, usually with aspect-preserving resize/pad and 14-pixel alignment; the default 1024-token budget includes separators. Extra-wide images are directly resized when `vision_max_wh_ratio` is triggered. | **DeepSeek-ViT is trained from scratch.**<br>① **SigLIP sigmoid contrastive loss**: **47B image-text pairs**, alt-text, up to 224² pixels.<br>② Attach a **4B auxiliary LLM** for **236B tokens of NTP** using captions, alt-text, charts, and OCR at 544²–1344² resolution; then discard the auxiliary LLM.<br>③ Train the main model on **45T all-modality tokens**: initially freeze the visual tower, training only its final Norm and projector; unfreeze the tower at a smaller LR during LR decay. Main-training LR is $2.6×10^{-4}$, decaying to $2.6×10^{-5}$ between 28T and 40T tokens.<br>④ **SFT→RL→OPD**, including multimodal tasks. | Overlap feature All-Gather with computation during contrastive training. Schedule visual encoding and main-model training in stages, balance image assignment, and avoid duplicate loading. Multimodal RL transfers images incrementally and caches CPU decoding/preprocessing for reuse by rollout and training. |
| **Muse Spark Open** | **Perception Encoder, 1.8B, 50 layers, width 1536.**<br>**Attention: MHA, Q/K/V = 16/16/16 heads, head dimension 96**; non-causal **32×32-patch local windows**. Layers **4, 8, …, 48, 50** use global attention.<br>FFN: standard two-layer **GELU MLP, 1536→8960→1536**; pre-LayerNorm, plus input and final encoder LayerNorm.<br>Projector: after merging, **6144→4096→4096→8192**, GELU after the first two layers, and final RMSNorm without scaling. | **Overall: $2×28×28$ for video; $1×28×28$ for static images.**<br>**Patch / compression:** flatten $2×14×14$ RGB pixel blocks + Linear, **1176→1536**; final LayerNorm followed by spatial $2×2$ pixel-unshuffle, **1536→6144, $N→N/4$**. Two frames are fused at the input; static images are replicated along time.<br>**Position:** bilinearly interpolated learned 32×32 spatial table + visual 2D axial RoPE, θ=10000.<br>**Dynamic resolution: yes.** Resize according to aspect ratio and token budget, with 28-pixel alignment. Pack variable-length patches; `cu_seqlens` separates images/temporal slices, with additional window segmentation in local layers. | — | — |
| **MiMo-V2.6-Pro** | **MiMo-ViT, 681M, 28 layers, width 1280.**<br>**Attention: GQA, Q/K/V = 32/8/8 heads, head dimension 64**. The 24 local layers use bidirectional SWA with sinks, covering 64 positions on either side in the one-dimensional sequence and switching row/column serialization by group. Global layers: **[0,9,18,27], zero-indexed**.<br>FFN: **SwiGLU, 1280→4608→1280**; RMSNorm within blocks.<br>Projector: LayerNorm followed by spatial $2×2$ concatenation, **5120→5120→6144, GELU**; separate merger. | **Overall: $2×32×32$ along the model's input time axis; final temporal factor is 1.**<br>**Patch / compression:** Conv3d-equivalent projection, kernel=stride=$2×16×16$, executable as a Linear operation with reshaped weights; final LayerNorm + spatial $2×2$ concatenation + MLP, **$N→N/4$**. No additional temporal pooling. `temporal_compression_ratio=1` must not introduce another temporal factor; `temporal_patch_size=2` determines the input block's temporal span.<br>**Position:** no absolute-position table; visual 2D axial RoPE.<br>**Dynamic resolution: yes.** Variable grids + `cu_seqlens` separate images/temporal slices; local layers reorder rows/columns. A pixel budget applies. | ① Separate visual pretraining: randomly initialize the ViT and attach a small pretrained LLM, which remains trainable. Use only multimodal-understanding cross-entropy/NTP, with over **4T image tokens**.<br>② The Pro main model first trains on **27T text tokens**, then adds the internally pretrained ViT for joint training on **3T all-modality tokens**; 30T total. Pretraining uses AdamW.<br>③ Mid-training with visual trajectories: 256K→1M context, switching the matrix optimizer to Muown.<br>④ Short SFT→mixed-task RL including visual tasks yields **Pro-RL**; MOPD2 then yields **Pro-MOPD**. | Replicate the visual encoder within each TP group; first balance images through data parallelism, then redistribute embeddings to the ranks holding the corresponding tokens. Pixels stay in distributed KV storage and are fetched only for encoding. Multi-turn tasks transfer only newly added multimodal payloads. |
| **Thinking Machines Inkling** | **HMLP, 83.74M, four stages of local folding and linear projection, output width 6144.**<br>Local projections: the first three HMLP stages are **75→128, 512→320, 5120→4800**, each using **Linear→RMSNorm→GELU**.<br>Projector: the final HMLP stage is **9600→6144, final RMSNorm**. | **Overall: $2×40×40$ for video; $1×40×40$ for static images.**<br>**Patch / compression:** the first three stages concatenate spatial $5×5$, $2×2$, and $4×4$ neighborhoods into channels, each followed by a linear projection. The spatial grid becomes $40×40→8×8→4×4→1×1$. The fourth stage concatenates two temporal positions and projects them into one 6144-dimensional token. Static images are replicated along time, which does not count as actual frame compression.<br>**Position:** no absolute-position table/RoPE or inter-patch attention inside HMLP.<br>**Dynamic resolution: yes.** Split the input into 40×40 blocks and pad boundaries. Optional aspect-preserving Lanczos rescaling; official guidance recommends 40–4096 pixels per side. | — | — |
| **GLM-5.3-Flash** | **ViT, 404.2M, 24 layers, width 1024.**<br>**Attention: MHA, Q/K/V = 16/16/16 heads, head dimension 64**; non-causal global attention over the spatial grid; per-head **QK RMSNorm**.<br>FFN: **SwiGLU, 1024→4096→1024**, with bias; **gate clipped only above at 10, up clipped to [-10,10]**; pre-RMSNorm and final encoder RMSNorm, $\epsilon=10^{-5}$.<br>Projector: after spatial convolution, **4096→4096, LayerNorm+GELU, then clipped SwiGLU with intermediate width 10240→4096**; separate projector and final spatial convolution total **159.4M**. | **Overall: $2×28×28$ along the model's input time axis; temporal factor after patchification is 1.**<br>**Patch / compression:** Conv3d kernel=stride=$2×14×14$; final **Conv2d, 1024→4096, kernel=stride=2**, halves height and width, **$N→N/4$**. No additional temporal pooling. Spatial compression uses a learned convolution.<br>**Position:** no absolute-position table; visual 2D axial RoPE.<br>**Dynamic resolution: yes.** `smart_resize` selects a canvas under a budget, applies aspect-preserving bicubic resizing and right/bottom padding, with 28-pixel alignment. Variable-length patches + `cu_seqlens` separate images/temporal slices. Static images are replicated along time at the input; deployment can override the budget. | — | — |
| **dots3-note-prev** | **MoE ViT, 7B total parameters, 1.2B active parameters, 42 layers, width 1536.** Estimates from the released implementation: **6.80B total / 1.216B active per token**.<br>**Attention: MHA, Q/K/V = 24/24/24 heads, head dimension 64**; non-causal global attention within each frame; **QK RMSNorm**.<br>FFN: first **25 layers use dense SwiGLU, 1536→4224→1536**; final **17 layers use visual MoE**, with expert counts **4,8,…,64,64**, **Top-2**, and **1536→2112→1536 SwiGLU** per expert. Pre-RMSNorm and final encoder RMSNorm.<br>Projector: LayerNorm followed by spatial $2×2$ concatenation, **6144→6144→5120, GELU**; separate adapter adds **69.2M**. | **Overall: $1×28×28$; no temporal compression.**<br>**Patch / compression:** Conv2d, $1×14×14$. The initial `pre_pixel_shuffle` only performs $2×2$ block-major reordering, keeping token count unchanged. Spatial $2×2$ concatenation occurs only in the adapter after all 42 layers, **$N→N/4$**. Images/videos are encoded frame by frame, with no temporal pooling.<br>**Position:** no learned absolute-position table; visual 2D axial RoPE, with spatial coordinates repeated per frame.<br>**Dynamic resolution: yes.** Variable `grid_thw`, variable-length patches, and `cu_seqlens` separating images/temporal slices. | — | — |

<!-- vit-details -->

## SigLIP 2

If I had to name the most influential vision encoder of the past two years, it would be SigLIP 2. It perfectly illustrates a modern classic in visual learning: combine the best ideas from everywhere. (For a related discussion, see my earlier blog, “From VAE to RAE: Generative Latent Representations That Want It All.”)

Let's focus on the widely used So400m version.

### 1. Architecture

- Standard ViT, approximately 400M visual-tower parameters, 27 layers, hidden width 1152.
- Attention: MHA, Q/K/V = 16/16/16 heads, head dimension 72; non-causal global attention.
- FFN: GELU, 1152→4304→1152; LayerNorm and multihead attention pooling. A learnable query cross-attends to all patches in the final layer, replacing mean pooling or a CLS token.
- Text encoder: also 27 layers, hidden width 1152; text length 64, Gemma 256K tokenizer.
- The LocCa decoder, EMA teacher, and auxiliary heads are used during training and discarded afterward.

### 2. Patches / Resolution / Position Encoding

- Patches: initial pretraining uses 16×16 patches and 256² images, producing 256 visual tokens; patch tokens are not compressed.
- Fixed resolution: Patch14 supports 224² and 384²; Patch16 supports 256², 384², and 512².
- Position encoding: learned absolute position embeddings, without RoPE.
- Dynamic resolution: NaFlex preserves the original aspect ratio as much as possible, adapts position embeddings through bilinear interpolation, and supports token budgets of 128/256/576/784/1024.

![SigLIP 2](assets/siglip2.png)

### 3. Training

Here is a quick look at SigLIP 2's varied losses.

#### ① SigLIP + LocCa (throughout training, loss ratio 1:1)

- Sigmoid loss: independently classify every image-text pair in the batch; matched pairs are positive and all others negative.
- LocCa: add a cross-attention decoder over ViT patch features for three generation tasks:
    - Image → Caption: generate a description of the full image.
    - Region Caption → BBox: predict coordinates from a region description.
    - BBox → Region Caption: generate a region description from coordinates.
- Training data: extract phrases from alt-text and use an open-vocabulary detector to generate region-box pseudo-labels automatically.
- Training procedure: share one ViT forward pass per image, then run three additional decoder forward passes. Captioning uses non-causal parallel token prediction with 50% probability.
- The decoder is used only for training and discarded at inference.

#### ② SILC / TIPS (added for the final 20%)

Use DINO/iBOT-style EMA teacher-student self-supervision:

- SILC (loss weight 1.0): the teacher sees one global view; the student sees eight local views. An additional MLP head matches global semantic distributions to improve local/global representation consistency.
- TIPS (loss weight 0.25): the teacher sees the complete global view. The student randomly masks 50% of patch embeddings and predicts the teacher features at the corresponding positions, rather than reconstructing RGB.
- The teacher is updated by EMA of the student, without backpropagation. SigLIP/LocCa still use the original images; self-supervision uses additional augmented images.

The total loss for So400m during the final 20% is:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{SigLIP}}+\mathcal{L}_{\mathrm{LocCa}}+\mathcal{L}_{\mathrm{SILC}}+0.25\mathcal{L}_{\mathrm{TIPS}}
$$

#### ③ Resolution Adaptation (two independent branches)

FixRes branches from the checkpoint at 95% of training.

- Increase image resolution and interpolate the learned position embeddings in 2D, for example 16×16 → 24×24.
- When switching Patch16 → Patch14, use PI-Resize (pseudoinverse) to transform patch-embedding weights, preserving projection outputs as closely as possible.
- Continue the remaining 5% of training at the target resolution with all losses, saving separate checkpoints for each resolution.

NaFlex branches from the 90% checkpoint.

- Preserve the original aspect ratio as much as possible; resize to multiples of the patch size without forcing square images.
- Sample a token budget from `{128, 256, 576, 784, 1024}` for each batch.
- Bilinearly interpolate the learned 16×16 position embeddings to the target patch grid; mask padding in attention.
- Extend the final-stage LR schedule by 3.75×; retain only SigLIP + LocCa, without SILC/TIPS.

#### ④ ACID: Data-Selection Distillation for Smaller Models

So400m can subsequently serve as the teacher for ViT-B through ACID distillation.

- First train So400m on another 1B high-quality samples.
- Teacher and student compute learnability for candidate samples, prioritizing those the teacher has mastered but the student has not.
- Select 32K samples from a 64K super-batch; continue student training for 4B samples.
- Use only sigmoid loss, without directly distilling teacher logits or features.

## Perception Encoder (PE-G)

Unlike SigLIP 2's combination of multiple pretraining losses, Perception Encoder (PE) primarily relies on large-scale CLIP contrastive learning. It finds powerful general-purpose visual features in intermediate layers, then transfers those capabilities to the output layer through alignment tuning.

### 1. Architecture

- Standard ViT, 1.88B visual-tower parameters, 50 layers, hidden width 1536.
- Attention: MHA, Q/K/V = 16/16/16 heads, head dimension 96; non-causal global attention.
- FFN: 1536→8960→1536; eight-head attention pooling, without a CLS token.
- Text encoder: 0.47B, 24 layers, hidden width 1280; 20 heads, FFN width 5120, text length 72; image-text contrastive dimension 1280.

### 2. Patches / Resolution / Position Encoding

- Patches: 14×14; the final 448² images correspond to 1024 visual tokens, without compressing patch features.
- Position encoding: 2D RoPE + learned absolute position embeddings; the latter support interpolation.
- Progressive resolution: 98²→154²→224²→336²→448².

### 3. Training

#### ① CLIP Contrastive Pretraining

- Loss: conventional CLIP bidirectional softmax contrastive loss, rather than SigLIP's sigmoid loss.
- Data: 5.4B image-text pairs filtered by MetaCLIP, totaling 86B samples seen.
- Optimization: LAMB, batch size 131K, LR 2e-3, 656K training steps.
- Augmentation: random crop (0.08–1.0), color jitter, and horizontal flip.
- Mask regularization: duplicate an additional 1/16 of the batch, randomly mask 40% of patches, and align masked/unmasked features through cosine similarity, with a gradient path separate from the CLIP loss.

#### ② Video Finetuning

- First cool down using 50M images, then finetune on 22M recaptioned videos.
- Video captions: combine video-captioner output, frame captions, and metadata, then integrate them with Llama 3.3 70B.
- Uniformly sample eight frames per video, encode each with the same ViT, and mean-pool them into a video embedding.
- Continue using CLIP contrastive loss, without adding temporal attention.

#### ③ Intermediate Layers

PE-G's best visual features are not necessarily in its final layer:

- OCR: layer 38; VQA: layer 47.
- Detection / depth: layer 39; tracking: layer 32.
- Global tokens emerge from layer 33. The final 18 layers increasingly aggregate global semantics, causing some loss of local spatial information.

The authors therefore branch PE-core-G into two alignment variants:

- PE-lang-G: retain the first 47 layers and discard the last three; attach Llama 3.2 3B + an MLP projector, then jointly finetune with CE loss on 70M multimodal samples to strengthen language understanding, including OCR/VQA.
- PE-spatial-G: jointly distill semantic features from layer 41 of frozen PE-core-G and spatial correspondences from SAM 2.1 mask logits. Train on 300M samples with 75% masking, giving the final layer stronger dense features.

## Encoder-Free

Finally, let me put in a word for the Encoder-Free approach we have been advocating! Excitingly, while I was writing this blog, Microsoft released its latest 3T model, MAI-Base-2, an Encoder-Free VLM!

The core idea is to remove the separate deep vision encoder and use a lightweight patchify projection for the input. Examples include NEO, NEO-ov, SenseNova-U1, SenseNova-U1.5, Gemma4-12B, and Inking. I also recommend exploring Haiwen Diao's publication list: a long-time pioneer of native modeling.

What are the benefits?

![Encoder-Free](assets/encoder-free.png)

My earlier blog, “Why Do We Want to Remove the Vision Encoder?” (you can find it on my homepage), also discussed a few small infrastructure benefits. Ultimately, my taste has always been “removing inductive bias as much as possible to let the model architecture take its natural and intended form.” If it works without the encoder, why not?
