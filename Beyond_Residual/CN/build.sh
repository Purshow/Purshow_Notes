#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source_file='Beyond_Residual_CN.md'
output_name='Beyond_Residual_CN'
mkdir -p .build
pandoc "$source_file" --from=markdown+tex_math_dollars+tex_math_single_backslash --standalone --pdf-engine=xelatex --variable=documentclass:ctexart --variable=classoption:fontset=none --variable=fontsize:11pt --variable=papersize:a4 --variable=geometry:margin=22mm,headheight=15pt --include-in-header=pdf-header.tex --lua-filter=cover.lua --output="$output_name.tex"
for pass in 1 2; do
  if ! xelatex -interaction=nonstopmode -halt-on-error -output-directory=.build "$output_name.tex" > .build/xelatex-output.txt 2>&1; then
    cat .build/xelatex-output.txt
    exit 1
  fi
done
cp ".build/$output_name.pdf" "$output_name.pdf"
printf '已生成：%s.pdf\n' "$output_name"
