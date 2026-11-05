#!/usr/bin/env bash
# The dome-shaped head array: 980 ppm.
#
#   bash scripts/paper/dome_head_array.sh [outdir]
#
# A lattice of 3,987 cubes of 12 mm fills eleven shells of a 230 mm hemisphere around
# a head bore. The optimizer seeds it by least squares and runs three stages with every
# freedom open, mutual demagnetization and the mean field pinned at 40 mT. Results are
# read on the 367,685 pinned points of data/head_roi.npz. About 15 hours on a 32 GB GPU.
# SMOKE=1 runs each stage for 20 epochs.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-runs/dome_head_array}"
# 20 epochs cannot resolve the lattice's overlaps, so a smoke run keeps designs with overlaps.
[ "${SMOKE:-0}" = 1 ] && export STAGE_EPOCHS="20 20 20" ALLOW_OVERLAPS=1
PROBLEM="--config configs/headscanner_geometry.cfg --geometry head --head-outer-mm 230 --head-zmax-mm 230
         --fov-resolution-mm 12 --train-boundary-points 2000
         --band-weight 1e7 --band-lower-t 0.040 --band-upper-t 0.040 --evaluation-points data/head_roi.npz"
OUT="$OUT" PROBLEM="$(echo $PROBLEM)" bash scripts/stages/all_freedoms_three_stages.sh
echo "final design: $OUT/final.pt"
