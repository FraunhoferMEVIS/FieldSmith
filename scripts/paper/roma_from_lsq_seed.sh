#!/usr/bin/env bash
# ROMA after least-squares initialization: 521 ppm after Stage I, 321 ppm in the end.
#
#   bash scripts/paper/roma_from_lsq_seed.sh [outdir]
#
# Stage I turns every magnet in its plane at the published seats, in 14 restarts from
# the least-squares seed at a mean field of 45.00 mT. Stages II and III open the cube
# tilts, then the centres within boxes of 3/1, 5/2, 7/3 and 9/4 mm around the seats,
# at 45.17 mT. About 35 hours on one RTX 5090. SMOKE=1 runs a short version: two restarts,
# the first box rung, 20 epochs per stage.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN="${1:-runs/roma_from_lsq_seed}"
[ -f data/roma/roma_lsq_inplane_start.pt ] && [ -f data/roma/roma_true.pt ] || ${PYTHON:-python3} scripts/build_roma_inputs.py
RESTARTS="4e-3:20:40000 5e-4:20:40000 5e-4:20:40000 1.5e-4:20:30000 1.5e-4:20:30000 1.5e-4:20:30000
          1.5e-4:20:30000 1.5e-4:20:30000 1.5e-4:40:30000 1.5e-4:40:30000 1.5e-4:40:30000 1.5e-4:40:30000
          1.5e-4:40:30000 1.5e-4:40:30000"
if [ "${SMOKE:-0}" = 1 ]; then
  RESTARTS="4e-3:20:20 5e-4:20:20"; export STAGE_EPOCHS="20 20 20 20" RUNG_EPOCHS=20 BOX_LADDER="5/2" EVERY=10
fi
OUT="$RUN/stage1_inplane" INIT=data/roma/roma_lsq_inplane_start.pt FIELD_T=0.045 FOV_MM=5 \
  RESTARTS="$(echo $RESTARTS)" bash scripts/stages/inplane_restarts.sh
OUT="$RUN/stages2_3" SEED="$RUN/stage1_inplane/best.pt" SEATS=data/roma/roma_true.pt FIELD_T=0.04517 FOV_MM=6 \
  bash scripts/stages/free_tilts_then_positions.sh
[ "${DRY:-0}" = 1 ] && exit 0
cp "$RUN/stages2_3/final.pt" "$RUN/final.pt"
echo "final design: $RUN/final.pt"
