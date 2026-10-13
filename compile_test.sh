#!/bin/bash
# Compile main.tex and si.tex in three configurations; report errors and unresolved \FM keys.
# Usage: TEX=manuscript MACROS=path/to/paper_macros.tex SYN=/tmp/fmout FIG1=figs/fig1_model.pdf \
#        B=/tmp/build bash tests/compile_test.sh
# real = your current macros; syn = synthetic numbers/figures/tables; synT = syn with every switch on.
set -u
TEX=${TEX:-manuscript}; MACROS=${MACROS:-paper_macros.tex}; SYN=${SYN:-/tmp/fmout}
FIG1=${FIG1:-figs/fig1_model.pdf}; B=${B:-/tmp/fm_build}; rm -rf $B; mkdir -p $B/real $B/syn $B/synT
cp $TEX/*.tex $B/real/; cp $MACROS $B/real/
for d in syn synT; do cp $TEX/*.tex $B/$d/; cp $SYN/paper_macros.tex $B/$d/; mkdir -p $B/$d/figs
  cp $SYN/*.pdf $B/$d/figs/; cp -r $SYN/tables $B/$d/; cp $FIG1 $B/$d/figs/fig1_model.pdf; done
for f in main.tex si.tex; do python3 - $B/synT/$f <<'PY'
import sys,re
f=sys.argv[1]; s=open(f).read()
for sw in re.findall(r"\\newif\\if(\w+)", s):
    s=s.replace("\\begin{document}", f"\\{sw}true\n\\begin{{document}}",1)
open(f,"w").write(s)
PY
done
for d in real syn synT; do for f in main si; do (cd $B/$d && pdflatex -interaction=nonstopmode -halt-on-error $f.tex >/dev/null 2>&1; pdflatex -interaction=nonstopmode -halt-on-error $f.tex > $f.stdout 2>&1; rc=$?
 echo "$d/$f: exit $rc pages $(pdfinfo $f.pdf 2>/dev/null | awk '/Pages/{print $2}') undef-refs $(grep -c 'Reference.*undefined' $f.log) overfull $(grep -c 'Overfull' $f.log) unresolved-FM $(pdftotext $f.pdf - 2>/dev/null | grep -o '??[A-Za-z_0-9.,]*' | sort -u | wc -l) TODO $(pdftotext $f.pdf - 2>/dev/null | grep -o 'TODO' | wc -l)"); done; done
