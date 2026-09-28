#!/bin/bash
# Build the AAMAS 2027 submission (paper/main.pdf) and the supplementary material (paper/supplement.pdf).
set -e
cd "$(dirname "$0")"
for doc in main supplement; do
  [ -f $doc.tex ] || continue
  pdflatex -interaction=nonstopmode -halt-on-error $doc.tex > /dev/null
  bibtex $doc > /dev/null || true
  pdflatex -interaction=nonstopmode -halt-on-error $doc.tex > /dev/null
  pdflatex -interaction=nonstopmode -halt-on-error $doc.tex > $doc.build.log
  grep -E "Warning.*(undefined|multiply)|Overfull|LaTeX Error" $doc.log | head -20 || true
  echo "$doc.pdf: $(pdfinfo $doc.pdf 2>/dev/null | grep Pages)"
done
