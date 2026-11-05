#!/usr/bin/env bash
# In-plane restarts: the angle-only optimization run several times in a row, each
# restart from the best design so far with a fresh optimizer state. Magnets turn in
# their plane only and every centre stays in its seat.
#
#   OUT=runs/x INIT=design.pt FIELD_T=0.045 RESTARTS="4e-3:20:40000 5e-4:20:40000" \
#   bash scripts/stages/inplane_restarts.sh
#
# RESTARTS has one "step:sharpness:epochs" entry per restart. CONFIG picks the field
# model, FRAME the body frame, LOCK=1 holds bodies that are not cubes in their plane.
# The best design of the chain is copied to $OUT/best.pt. DRY=1 prints the commands.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:?set OUT}"; INIT="${INIT:?set INIT}"; FIELD_T="${FIELD_T:?set FIELD_T}"; RESTARTS="${RESTARTS:?set RESTARTS}"
CONFIG="${CONFIG:-configs/roma_mixed200_ideal.cfg}"; FRAME="${FRAME:-seated}"; LOCK="${LOCK:-1}"
FOV_MM="${FOV_MM:-5}"; CHUNK="${CHUNK:-1024}"; EVERY="${EVERY:-2000}"; EVAL="${EVAL:-data/roma/roma_roi_200.npz}"
PYTHON="${PYTHON:-python3}"; DRY="${DRY:-0}"
mkdir -p "$OUT"; LOG="$OUT/restarts.log"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }
best_of() { grep -F '[truth]' "$1" | awk '{print $4}' | sort -n | head -1; }
say "in-plane restarts in $OUT: from $INIT, field $FIELD_T T"
k=0
for ENTRY in $RESTARTS; do
  k=$((k + 1))
  IFS=: read -r LR KAP EPO <<< "$ENTRY"
  ARGS=(--config "$CONFIG" --geometry dome --double-precision --init-design "$INIT" --target 1,0,0
        --frame-rule "$FRAME" --fov-resolution-mm "$FOV_MM" --chunk-points "$CHUNK"
        --roi-shape sphere --roi-radius-mm 100 --roi-zmin-mm -100 --train-boundary-points 3000
        --num-epochs "$EPO" --lr "$LR" --polar-lr 0
        --alpha-auto "$KAP" --alpha-max 1e6 --rms-weight 0.2
        --band-weight 1e7 --band-lower-t "$FIELD_T" --band-upper-t "$FIELD_T"
        --evaluate-every "$EVERY" --evaluation-points "$EVAL"
        --best-design "$OUT/restart${k}_best.pt" --log-each-n-epochs 1000 --wandb-mode disabled)
  [ "$LOCK" = 1 ] && ARGS+=(--lock-elongated-polar)
  say "restart $k: $EPO epochs, step $LR, sharpness $KAP, from $INIT"
  if [ "$DRY" = 1 ]; then echo "$PYTHON -u scripts/run_dome_optimisation.py ${ARGS[*]}"; INIT="$OUT/restart${k}_best.pt"; continue; fi
  $PYTHON -u scripts/run_dome_optimisation.py "${ARGS[@]}" > "$OUT/restart${k}.log" 2>&1
  rc=$?
  [ -f "$OUT/restart${k}_best.pt" ] || { say "restart $k saved no design (rc $rc), stopping"; exit 1; }
  this=$(best_of "$OUT/restart${k}.log")
  say "restart $k done (rc $rc), best $this ppm"
  # Carry the best design of the chain, not the last restart's.
  if [ -z "${BEST_PPM:-}" ] || [ "${this:-999999}" -lt "$BEST_PPM" ] 2>/dev/null; then
    BEST_PPM="$this"; BEST_DESIGN="$OUT/restart${k}_best.pt"
  else
    say "  restart $k did not beat $BEST_PPM ppm, the chain keeps $BEST_DESIGN"
  fi
  INIT="$BEST_DESIGN"
done
[ "$DRY" = 1 ] && exit 0
cp "$INIT" "$OUT/best.pt"
say "best of the chain: ${BEST_PPM:-?} ppm, saved as $OUT/best.pt"
