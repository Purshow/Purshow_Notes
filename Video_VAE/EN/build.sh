#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p .build
pandoc Video_VAE_EN.md \
  --from=markdown+tex_math_dollars+tex_math_single_backslash+raw_html \
  --standalone --variable=documentclass:article \
  --variable=fontsize:11pt --variable=colorlinks:true --variable=papersize:a4 \
  --variable=geometry:landscape,margin=18mm,headheight=15pt \
  --include-in-header=pdf-header.tex --lua-filter=pdf-filter.lua \
  --syntax-highlighting=none --output=Video_VAE_EN.tex
for pass in 1 2; do
  if ! xelatex -interaction=nonstopmode -halt-on-error \
      -output-directory=.build Video_VAE_EN.tex > .build/xelatex-output.txt 2>&1; then
    cat .build/xelatex-output.txt
    exit 1
  fi
done
cp .build/Video_VAE_EN.pdf Video_VAE_EN.pdf
printf 'Generated: Video_VAE_EN.pdf\n'
