#!/usr/bin/env bash
# Render notes/*.md to pdf/*.pdf: pandoc (Markdown -> HTML, math via KaTeX) + headless Chromium (HTML -> PDF).
# No LaTeX needed. KaTeX is fetched once with `npm pack` into .cache/ (gitignored).
# Usage: ./build_pdf.sh [notes/0X_*.md ...]
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/opt/pw-browsers/chromium-1194/chrome-linux/chrome}"
KATEX_VER=0.16.11
KATEX_DIR="$PWD/.cache/katex-$KATEX_VER"
if [ ! -f "$KATEX_DIR/katex.min.js" ]; then
  tmp=$(mktemp -d); (cd "$tmp" && npm pack "katex@$KATEX_VER" >/dev/null 2>&1 && tar xzf "katex-$KATEX_VER.tgz")
  mkdir -p "$KATEX_DIR"; cp -r "$tmp/package/dist/." "$KATEX_DIR/"; rm -rf "$tmp"
fi
mkdir -p pdf
files=("$@"); [ ${#files[@]} -eq 0 ] && files=(notes/*.md)
for md in "${files[@]}"; do
  name=$(basename "$md" .md)
  html="$PWD/.cache/$name.html"
  pandoc "$md" -s --katex="file://$KATEX_DIR/" --metadata title="$name" -H pdf/style.css.html -o "$html"
  "$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer --allow-file-access-from-files \
    --virtual-time-budget=20000 --print-to-pdf="pdf/$name.pdf" "file://$html" 2>/dev/null
  echo "pdf/$name.pdf"
done
