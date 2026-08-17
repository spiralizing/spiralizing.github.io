#!/usr/bin/env bash
# Rebuild the /Research/ triangle figure from its TikZ source.
#   latex -> DVI -> dvisvgm (real <text>, embedded WOFF2) -> theme_svg.py
# Writes _assets/research_triangle.svg and re-splices it into Research.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cp "$here/research_triangle.tex" "$work/"
(cd "$work" && latex -interaction=nonstopmode -halt-on-error research_triangle.tex >/dev/null)
# --exact-bbox: tight box from the glyph outlines, not the (padded) font metrics
# --font-format=woff2: keep <text> selectable instead of flattening it to paths
(cd "$work" && dvisvgm --exact-bbox --font-format=woff2 --precision=4 \
    -o raw.svg research_triangle.dvi >/dev/null 2>&1)

python3 "$here/theme_svg.py" "$work/raw.svg" \
    "$root/_assets/research_triangle.svg" "$root/Research.md"
