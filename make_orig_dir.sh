#!/bin/bash
# tests/make_orig_dir.sh — rebuild the code AS IT WAS BEFORE the Paper 1 changes (for ORIG_DIR).
# Run it from the code folder that already has the Paper 1 files:
#     bash tests/make_orig_dir.sh paper1/patches                 # -> $HOME/FlowMergence_before_paper1
#     bash tests/make_orig_dir.sh paper1/patches /some/other/dir
# It copies every *.py and reverses the Paper 1 patches of the modules the tests compare.
# This is a copy of CODE, not of results: the data root is not touched.
set -euo pipefail
P="$(cd "${1:?usage: make_orig_dir.sh <patches dir> [dest]}" && pwd)"
D="${2:-$HOME/FlowMergence_before_paper1}"
mkdir -p "$D"
cp ./*.py "$D"/
cd "$D"
for f in params dynamics metrics config paths; do
    patch -R -s "$f.py" < "$P/$f.py.diff" || { echo "could not reverse $f.py: is $f.py exactly the Paper 1 version?"; exit 1; }
done
rm -f hubs.py meanfield.py cost_calibration.py           # new in Paper 1
python -c "import params, config; print('old code OK:', len(config.EXPERIMENTS), 'experiments')"
echo "ORIG_DIR=$D"
