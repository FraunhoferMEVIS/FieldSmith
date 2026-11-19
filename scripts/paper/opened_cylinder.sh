#!/usr/bin/env bash
# The ROMA cylinder with a 90 degree sector removed along its full length: 672 ppm.
#
#   bash scripts/paper/opened_cylinder.sh [outdir]
#
# The 1,726 remaining magnets start from their in-plane least-squares seed and are
# optimized in three stages with every freedom open: cubes turn and tilt, bars stay
# in their plane, centres move within their ring planes. The mean field is pinned at
# 20 mT. About 8 hours on one RTX 5090. SMOKE=1 runs each stage for 20 epochs.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-runs/opened_cylinder}"
[ -f data/roma/roma_open90_planar_seed.pt ] || ${PYTHON:-python3} scripts/build_roma_inputs.py
[ "${SMOKE:-0}" = 1 ] && export STAGE_EPOCHS="20 20 20"
# The explicit axial bound keeps the per-axis arithmetic the reported run used.
PROBLEM="--config configs/roma_mixed200_ideal.cfg --geometry dome --target 1,0,0 --frame-rule seated
         --lock-elongated-polar --fov-resolution-mm 6 --chunk-points 1024 --roi-shape sphere --roi-radius-mm 100
         --roi-zmin-mm -100 --train-boundary-points 3000 --position-bound-axial-mm 2
         --band-weight 1e7 --band-lower-t 0.020 --band-upper-t 0.020 --evaluation-points data/roma/roma_roi_200.npz"
OUT="$OUT" SEED=data/roma/roma_open90_planar_seed.pt PROBLEM="$(echo $PROBLEM)" \
  bash scripts/stages/all_freedoms_three_stages.sh
echo "final design: $OUT/final.pt"
