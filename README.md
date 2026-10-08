# Purshow Notes

Personal research notes.

## Contents

- `Attention/CN`, `Attention/EN`: Attention Architectures and Mechanisms
- `Beyond_Residual/CN`, `Beyond_Residual/EN`: Beyond Residual Connection
- `Encoder-Free`: Why Removing the Encoder Is Better from an Infrastructure Perspective
- `MoE/CN`, `MoE/EN`: Mixture of Experts
- `OPD/CN`, `OPD/EN`: On-Policy Distillation
- `Ngram_Embedding/CN`, `Ngram_Embedding/EN`: Sparse Vocab Embeddings: Over Encoding, PLE, and Engram
- `Video_VAE/CN`, `Video_VAE/EN`: VAEs for Video Generation Models
- `WAM/CN`, `WAM/EN`: World Action Models
- `YOCO/CN`, `YOCO/EN`: YOCO: Cross-Layer KV Sharing

## Compile

```bash
bash Beyond_Residual/build.sh
bash YOCO/build.sh
bash Ngram_Embedding/build.sh
bash Video_VAE/build.sh
(cd Attention/CN && xelatex -interaction=nonstopmode -halt-on-error Attention.tex && xelatex -interaction=nonstopmode -halt-on-error Attention.tex)
(cd Attention/EN && xelatex -interaction=nonstopmode -halt-on-error Attention_EN.tex && xelatex -interaction=nonstopmode -halt-on-error Attention_EN.tex)
(cd MoE/CN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error MoE_CN.tex)
(cd MoE/EN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error MoE_EN.tex)
(cd OPD/CN && latexmk -xelatex -interaction=nonstopmode -halt-on-error OPD_CN.tex)
(cd OPD/EN && latexmk -xelatex -interaction=nonstopmode -halt-on-error OPD_EN.tex)
(cd WAM/CN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error WAM_CN.tex && xelatex -shell-escape -interaction=nonstopmode -halt-on-error WAM_CN.tex)
(cd WAM/EN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error WAM_EN.tex && xelatex -shell-escape -interaction=nonstopmode -halt-on-error WAM_EN.tex)
```
