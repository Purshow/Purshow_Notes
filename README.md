# Purshow Notes

Notes on contemporary MoE configurations and on-policy distillation.

## Contents

- `MoE/CN/`: Chinese MoE notes, source, and compiled PDF
- `MoE/EN/`: English MoE notes, source, and compiled PDF
- `OPD/CN/`: Chinese OPD source and compiled PDF
- `OPD/EN/`: English OPD source and compiled PDF

## Compile

```bash
cd MoE/CN
xelatex -shell-escape -interaction=nonstopmode -halt-on-error moe_pdf.tex

cd ../../OPD/EN
xelatex -interaction=nonstopmode -halt-on-error TML_OPD_notes.tex
xelatex -interaction=nonstopmode -halt-on-error TML_OPD_notes.tex
```
