# Purshow Notes

Personal research notes.

## Contents

- `MoE/CN`, `MoE/EN`: Mixture of Experts
- `OPD/CN`, `OPD/EN`: Online Packing and Dispatching
- `WAM/CN`: World Action Models

## Compile

```bash
(cd MoE/CN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error moe_pdf.tex)
(cd MoE/EN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error moe_en_pdf.tex)
(cd OPD/CN && latexmk -xelatex -interaction=nonstopmode -halt-on-error TML_OPD_notes.tex)
(cd OPD/EN && latexmk -xelatex -interaction=nonstopmode -halt-on-error TML_OPD_notes.tex)
(cd WAM/CN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error wam.tex && xelatex -shell-escape -interaction=nonstopmode -halt-on-error wam.tex)
```
