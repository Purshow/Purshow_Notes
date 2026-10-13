#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/muon-en.XXXXXX")"
trap 'rm -rf -- "$build_dir"' EXIT

pandoc Muon_EN.md \
  --from=markdown+tex_math_dollars+tex_math_single_backslash+autolink_bare_uris \
  --standalone --pdf-engine=xelatex \
  --variable=documentclass:article --metadata=lang:en \
  --variable=fontsize:10pt --variable=papersize:a4 \
  --variable=geometry:margin=20mm,headheight=15pt \
  --include-in-header=pdf-header.tex --lua-filter=pdf-filter.lua \
  --output=Muon_EN.tex

for pass in 1 2; do
  if ! xelatex -interaction=nonstopmode -halt-on-error \
    -output-directory="$build_dir" Muon_EN.tex \
    > "$build_dir/xelatex-output.txt" 2>&1; then
    cat "$build_dir/xelatex-output.txt"
    exit 1
  fi
done

if rg -n 'Overfull|Missing character' "$build_dir/Muon_EN.log"; then
  printf 'Layout warnings found; review the generated PDF.\n' >&2
fi
cp "$build_dir/Muon_EN.pdf" Muon_EN.pdf
printf 'Generated: Muon_EN.pdf\n'
