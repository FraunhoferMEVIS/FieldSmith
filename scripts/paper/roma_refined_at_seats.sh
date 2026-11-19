#!/usr/bin/env bash
# Refinement of the published ROMA angles at the published seats: 332 ppm.
#
#   bash scripts/paper/roma_refined_at_seats.sh [outdir]
#
# Seven restarts turn every magnet in its plane, centres fixed, with the full cuboid
# model at a mean field of 45.00 mT. The bars stay flat through a zero out-of-plane
# step, as in the reported run. About 16 hours on one RTX 5090. SMOKE=1 runs two
# restarts of 20 epochs.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-runs/roma_refined_at_seats}"
[ -f data/roma/roma_true.pt ] || ${PYTHON:-python3} scripts/build_roma_inputs.py
RESTARTS="1.5e-4:40:30000 1.5e-4:40:30000 1.5e-4:40:30000 1.5e-4:40:30000 1.5e-4:40:30000 1.5e-4:40:30000 1.5e-4:40:30000"
[ "${SMOKE:-0}" = 1 ] && { RESTARTS="1.5e-4:40:20 1.5e-4:40:20"; export EVERY=10; }
OUT="$OUT" INIT=data/roma/roma_true.pt FIELD_T=0.045 CONFIG=configs/roma_cuboid200_ideal.cfg FRAME=world LOCK=0 \
  FOV_MM=8 CHUNK=2048 RESTARTS="$RESTARTS" bash scripts/stages/inplane_restarts.sh
[ "${DRY:-0}" = 1 ] && exit 0
cp "$OUT/best.pt" "$OUT/final.pt"
echo "final design: $OUT/final.pt"
