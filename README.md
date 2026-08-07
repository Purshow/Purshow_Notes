# Purshow Notes

My personal notes.

## Contents

- `MoE/CN/`: Chinese MoE notes, source, and compiled PDF
- `MoE/EN/`: English MoE notes, source, and compiled PDF
- `OPD/CN/`: Chinese OPD source and compiled PDF
- `OPD/EN/`: English OPD source and compiled PDF

The EN editions were translated by GPT based on the corresponding CN editions.

## Compile

```bash
# Run these commands from the repository root.
(cd MoE/CN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error moe_pdf.tex)
(cd MoE/EN && xelatex -shell-escape -interaction=nonstopmode -halt-on-error moe_en_pdf.tex)

(cd OPD/CN && xelatex -interaction=nonstopmode -halt-on-error TML_OPD_notes.tex)
(cd OPD/CN && xelatex -interaction=nonstopmode -halt-on-error TML_OPD_notes.tex)
(cd OPD/EN && xelatex -interaction=nonstopmode -halt-on-error TML_OPD_notes.tex)
(cd OPD/EN && xelatex -interaction=nonstopmode -halt-on-error TML_OPD_notes.tex)
```
