# Purshow Notes

Personal research notes.

## Contents

- `Attention/CN`, `Attention/EN`: Attention Architectures and Mechanisms
- `Beyond_Residual/CN`, `Beyond_Residual/EN`: Beyond Residual Connection
- `Encoder-Free`: Why Removing the Encoder Is Better from an Infrastructure Perspective
- `MoE/CN`, `MoE/EN`: Mixture of Experts
- `OPD/CN`, `OPD/EN`: On-Policy Distillation
- `WAM/CN`, `WAM/EN`: World Action Models

## Compile

```bash
bash Beyond_Residual/build.sh
(cd Attention/CN && xelatex -interaction=nonstopmode -halt-on-error Attention.tex && xelatex -interaction=nonstopmode -halt-on-error Attention.tex)
(cd Attention/EN && xelatex -interaction=nonstopmode -halt-on-error Attention_EN.tex && xelatex -interaction=nonstopmode -halt-on-error Attention_EN.tex)
(cd MoE/CN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error MoE_CN.tex)
(cd MoE/EN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error MoE_EN.tex)
(cd OPD/CN && latexmk -xelatex -interaction=nonstopmode -halt-on-error OPD_CN.tex)
(cd OPD/EN && latexmk -xelatex -interaction=nonstopmode -halt-on-error OPD_EN.tex)
(cd WAM/CN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error WAM_CN.tex && xelatex -shell-escape -interaction=nonstopmode -halt-on-error WAM_CN.tex)
(cd WAM/EN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error WAM_EN.tex && xelatex -shell-escape -interaction=nonstopmode -halt-on-error WAM_EN.tex)
```

## Beyond Residual Connection

Chinese research notes and a complete English translation covering PreNorm, HC, mHC, iHC, Gated Residual, Attention Residuals, xHC, and VWN, organized into `Beyond_Residual/CN` and `Beyond_Residual/EN` like the Attention and MoE notes.

| Edition | Writing source | Reading copy |
| :--- | :--- | :--- |
| 中文 | [Beyond_Residual_CN.md](Beyond_Residual/CN/Beyond_Residual_CN.md) | [Beyond_Residual_CN.pdf](Beyond_Residual/CN/Beyond_Residual_CN.pdf) |
| English | [Beyond_Residual_EN.md](Beyond_Residual/EN/Beyond_Residual_EN.md) | [Beyond_Residual_EN.pdf](Beyond_Residual/EN/Beyond_Residual_EN.pdf) |

Each edition contains Markdown, generated LaTeX and PDF, `assets/`, `pdf-header.tex`, `cover.lua`, and a build script. The cover contains the title, eight-entry contents, and a six-row overview table. The iHC section stays on one page in both editions. The English edition preserves the original figures and includes translations of the Chinese screenshot text.

### Build

Run from the `Purshow_Notes` directory to compile both editions:

```bash
bash Beyond_Residual/build.sh
```

Or compile one edition:

```bash
bash Beyond_Residual/CN/build.sh
bash Beyond_Residual/EN/build.sh
```

Dependencies: Pandoc, XeLaTeX, and the TeX Live / macOS fonts configured in each edition's `pdf-header.tex`. The Chinese edition uses `ctexart`; the English edition uses `article` and the English Attention note's heading style.

Edit Markdown for content, `pdf-header.tex` for typography, and `cover.lua` for the cover and table layout. Each build regenerates LaTeX, runs XeLaTeX twice, and updates the PDF. Logs and intermediate files stay in the ignored `.build/` directories. Generated `.tex` files can also be compiled directly with XeLaTeX from their edition directory.

整理后的写作位置是 `Beyond_Residual/CN/` 和 `Beyond_Residual/EN/`。原来的 `Codebase/Beyond_Residual/` 工作目录及飞书原稿备份仍保留。中文版仅调整附件路径，英文版保留作者观点、数字、公式和语气；中文更新后需要同步更新英文稿。
