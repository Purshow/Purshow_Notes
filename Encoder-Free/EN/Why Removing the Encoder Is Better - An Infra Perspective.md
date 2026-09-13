# Why Removing the Encoder Is Better from an Infra Perspective

Over the past six months, the main theme of my research has become removing as much Inductive Bias as possible. Following @Haiwen, I have also become a staunch advocate of Encoder-Free models.

Most multimodal large language models (MLLMs) today use the LLaVA architecture: a Vision Encoder such as SigLIP first processes the image, a Projector maps the visual features into the LLM's representation space, and the resulting features are then merged with text tokens and fed into the LLM.

This design works very well. LLaVA practically ignited the multimodal boom (a belief I am especially proud of...), but at large training scales, its systems cost is not merely the addition of a ViT with a few hundred million parameters. More importantly, it inserts into an otherwise relatively regular LLM training graph a computation path with a different compute scale, input shape, and parallelization strategy.

PS: This article focuses on training scenarios in which the Vision Encoder must be executed online and end to end. If the Encoder is fully frozen and its visual features can be precomputed offline, some of the issues discussed below become much less significant.

---

## 1. The Vision Encoder Changes an Otherwise Regular Computation Graph

The backbone of a text-only LLM consists of a sequence of structurally similar Transformer Layers:

```text
Tokens → Embedding → Transformer Layer 1 → ... → Transformer Layer N
```

Text-only training is not perfectly regular either: packing and MoE routing both introduce variance. But the Transformer backbone itself is at least regular enough to make it easier to build a stable cost model and combine parallelization strategies such as DP, TP, PP, CP, and EP.

An Encoder-based MLLM adds a separate visual path:

```text
Text IDs → Embedding ──────────┐
                              ├→ Merge → LLM → Loss
Pixels → ViT → Projector ──────┘
```

Here, the ViT takes as input visual tokens produced by patchifying the image.

The ViT and the LLM are not two completely unrelated forms of computation. Both are usually built primarily from Transformer Blocks, but they can differ substantially in depth, hidden size, sequence length, input shape, and optimal parallel configuration. Therefore, even though they belong to the same end-to-end model, they are not necessarily well suited to the same execution strategy.

The Projector is not the focus of this article. In common architectures that use a shallow MLP Projector, its compute is typically much smaller than that of the Vision Encoder and the LLM. More complex Connectors such as Q-Former need to be evaluated separately, although few people seem to use them anymore.

---

## 2. The Compute Cost of Multimodal Samples Is Harder to Predict

For the dynamic-resolution Encoder discussed here, input resolution, the number of images, and the number of video frames directly affect the number of visual tokens. For example:

```text
Sample A: pure text
Sample B: one 448×448 image
Sample C: eight high-resolution images
Sample D: a long video
```

The compute cost of these samples contains at least two components:

```text
Cost(sample)
≈ C_encoder(N_encoder_visual_tokens)
  + C_llm(N_text_tokens + N_llm_visual_tokens)
```

Here, `N_encoder_visual_tokens` is the number of patch/frame tokens entering the Vision Encoder, while `N_llm_visual_tokens` is the number of visual tokens fed into the LLM after the Projector or token merging.

This creates two types of load imbalance:

- Different Data Parallel ranks may receive images with different resolutions, different numbers of images, or videos of different lengths. At synchronization points, everyone must wait for the slowest rank.
- Different micro-batches may carry different visual workloads, causing the service time of a Pipeline stage to fluctuate over time.

If the training data contains a mixture of text-only, single-image, multi-image, and video samples, the execution path also changes with the sample type: text-only samples do not execute the Vision Encoder, while video samples may introduce far more visual computation than the average.

In practice, systems can reduce this variance through various techniques, but doing so means that the sampler and scheduler must understand modality information. Batch size or text-token count alone cannot accurately describe the real workload. Even if all tokens entering the LLM are counted, the Encoder-side computation still needs to be included.

---

## 3. The Vision Encoder and the LLM Often Need Different Parallelization Strategies

First, consider Pipeline Parallelism. A common placement strategy is to colocate the Vision Encoder, Projector, and some LLM Layers on the first Pipeline stage:

```text
PP0: [Vision Encoder + Projector + LLM Layer 0..5]
PP1: [LLM Layer 6..17]
PP2: [LLM Layer 18..29]
PP3: [LLM Layer 30..41]
```

For a text-only LLM, the number of Layers on each stage can be adjusted so that the average execution times of the stages are as close as possible. In the layout above, however:

```text
T_PP0
= T_encoder(images, resolution, frames)
  + T_projector
  + T_llm_layer_0..5
```

`T_PP0` may not only be higher than the execution time of the other stages, but may also vary with the visual workload in each micro-batch. When PP0 becomes the bottleneck, downstream stages sit idle because they have no micro-batch to consume, creating additional pipeline bubbles or back-pressure.

Next, consider TP, CP, and model placement. A large MoE LLM may use high degrees of TP, CP, PP, and EP to accommodate its hidden size, long context, and expert parameter count. A Vision Encoder with only a few hundred million parameters, however, often does not need the same degree of partitioning:

- Excessive TP may make the per-GPU GEMMs of the Encoder too small, increasing the relative share of communication overhead.
- The LLM and Vision Encoder have different sequence lengths and memory pressure, so their optimal CP degrees are often different as well.
- Mainstream ViTs do not use MoE, so the LLM's EP does not apply to the ViT.

The training system therefore faces a trade-off. Of course, there now seem to be additional ways of handling this problem, but I will use a simplified description here:

```text
Option A: The Encoder and LLM share one parallel configuration
          The implementation is simpler, but the Encoder may be
          over-partitioned, or PP0 may become too heavy.

Option B: The Encoder and LLM each use their own parallel configuration
          Each side can be optimized separately. If their GPU groupings
          or tensor-partitioning schemes are incompatible, the outputs
          and gradients at the boundary require additional data-layout
          transformations.
```

In short, sharing one parallel configuration may sacrifice efficiency, while using separate configurations increases scheduling and communication complexity.

---

## 4. What Encoder-Free Solves---and What It Does Not

The systems value of Encoder-Free is that it removes the deep ViT, allowing the main visual modeling and text modeling workloads to reuse the same decoder stack. From an infra perspective, this brings three main benefits:

- It reduces parallel-configuration conflicts between the Encoder and the LLM.
- It reduces conditional branches in which only some samples execute the deep Encoder.
- It makes scheduling, profiling, and performance modeling more similar to existing LLM infrastructure.

Of course, differences in resolution, number of images, and number of video frames still cause different samples to produce different numbers of visual tokens. Variable sequence lengths and load differences across samples therefore do not disappear completely. The improvement is that these tokens no longer need to pass through a deep ViT with its own parallelization requirements first.
