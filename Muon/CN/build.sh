#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

# Dependencies: Pandoc, MacTeX 2026, and macOS Songti / Heiti fonts.
# The source, header, and layout filter are all stored beside this script.
for tool in pandoc xelatex; do
  if ! command -v "$tool" > /dev/null 2>&1; then
    printf '缺少依赖：%s，请安装并加入 PATH。\n' "$tool" >&2
    exit 1
  fi
done

output_name='Muon_CN'
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/muon-cn.XXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

pandoc "$output_name.md" \
  --from=markdown+tex_math_dollars \
  --standalone --pdf-engine=xelatex \
  --variable=documentclass:ctexart --variable=classoption:fontset=none \
  --variable=fontsize:11pt --variable=papersize:a4 \
  --variable=geometry:margin=21mm,headheight=15pt \
  --include-in-header=pdf-header.tex --lua-filter=pdf-filter.lua \
  --output="$build_dir/$output_name.tex"

for pass in 1 2; do
  if ! xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory="$build_dir" "$build_dir/$output_name.tex" \
    > "$build_dir/xelatex-output.txt" 2>&1; then
    cat "$build_dir/xelatex-output.txt" >&2
    exit 1
  fi
done

if grep -nE 'Overfull|Missing character' "$build_dir/$output_name.log"; then
  printf '发现排版警告，请检查生成的 PDF。\n' >&2
fi

cp "$build_dir/$output_name.tex" "$output_name.tex"
cp "$build_dir/$output_name.pdf" "$output_name.pdf"
printf '已生成：%s.pdf\n' "$output_name"
