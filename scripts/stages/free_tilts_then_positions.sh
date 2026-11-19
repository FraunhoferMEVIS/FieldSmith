#!/usr/bin/env bash
# Opens the remaining freedoms of a ROMA design that has finished its in-plane stage:
# first the cube tilts, then the centres, inside a box around the original seats that
# widens rung by rung. Every stage restarts from the previous best with a smaller step.
#
#   OUT=runs/x SEED=inplane_best.pt SEATS=data/roma/roma_true.pt FIELD_T=0.04517 \
#   bash scripts/stages/free_tilts_then_positions.sh
#
# Stage  freedom                        epochs  angle lr  tilt lr  centre lr  box xy / z
#   1    + cube tilts, seeded KICK_DEG   30k     1.5e-4    1.5e-4   -          -     bars stay flat
#   2    + centres                       60k     1.5e-4    1.5e-4   1e-5       3 / 1 mm around the seats
#   3    settle                          30k     5e-5      5e-5     3e-6       3 / 1
#   4    settle                          30k     1.5e-5    1.5e-5   1e-6       3 / 1
#   5+   for each rung of BOX_LADDER (default "5/2 7/3 9/4")
#        widen the box                   20k     7e-5      7e-5     5e-6       rung
#        settle                          20k     3.5e-5    3.5e-5   2.5e-6     rung
#
# STAGE_EPOCHS (four values) and RUNG_EPOCHS shorten a test run, and DRY=1 prints the
# commands. The final design is $OUT/final.pt.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:?set OUT}"; SEED="${SEED:?set SEED}"; SEATS="${SEATS:?set SEATS}"; FIELD_T="${FIELD_T:?set FIELD_T}"
CONFIG="${CONFIG:-configs/roma_mixed200_ideal.cfg}"; FOV_MM="${FOV_MM:-6}"; CHUNK="${CHUNK:-1024}"; KAPPA="${KAPPA:-40}"
BOX_XY="${BOX_XY:-3.0}"; BOX_Z="${BOX_Z:-1.0}"; BOX_LADDER="${BOX_LADDER-5/2 7/3 9/4}"
KICK_DEG="${KICK_DEG:-0.25}"; KICK_SEED="${KICK_SEED:-1}"; RUNG_EPOCHS="${RUNG_EPOCHS:-20000}"
EVERY="${EVERY:-2000}"; EVAL="${EVAL:-data/roma/roma_roi_200.npz}"; PYTHON="${PYTHON:-python3}"
DRY="${DRY:-0}"
read -r -a EP <<< "${STAGE_EPOCHS:-30000 60000 30000 30000}"
mkdir -p "$OUT"; LOG="$OUT/stages.log"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }
best_of() { grep -F '[truth]' "$1" | awk '{print $4}' | sort -n | head -1; }
say "tilts then positions in $OUT: seed $SEED, seats $SEATS, field $FIELD_T T, ladder [$BOX_LADDER]"

# stage  epochs  angle_lr  tilt_lr  centre_lr  box_xy  box_z   ("-" keeps the centres in their seats)
TABLE=("1 ${EP[0]} 1.5e-4 1.5e-4 0    - -"
       "2 ${EP[1]} 1.5e-4 1.5e-4 1e-5 $BOX_XY $BOX_Z"
       "3 ${EP[2]} 5e-5   5e-5   3e-6 $BOX_XY $BOX_Z"
       "4 ${EP[3]} 1.5e-5 1.5e-5 1e-6 $BOX_XY $BOX_Z")
N=4
for rung in $BOX_LADDER; do
  N=$((N + 1)); TABLE+=("$N $RUNG_EPOCHS 7e-5   7e-5   5e-6   ${rung%/*} ${rung#*/}")
  N=$((N + 1)); TABLE+=("$N $RUNG_EPOCHS 3.5e-5 3.5e-5 2.5e-6 ${rung%/*} ${rung#*/}")
done
# A flat, mirror-symmetric design has no gradient towards tilting, so the tilt stage
# starts from a small random tilt of the cubes.
INIT="$OUT/seed_kicked.pt"
say "seeding the cube tilts with $KICK_DEG deg of random scatter (seed $KICK_SEED)"
if [ "$DRY" != 1 ]; then
  $PYTHON scripts/stages/kick_tilt.py "$SEED" "$INIT" --sigma-deg "$KICK_DEG" --seed "$KICK_SEED" >> "$LOG" 2>&1 || { say "kick failed"; exit 1; }
fi
for row in "${TABLE[@]}"; do
  read -r N EPOCHS ALR TLR CLR XY Z <<< "$row"
  ARGS=(--config "$CONFIG" --geometry dome --double-precision --init-design "$INIT" --target 1,0,0
        --frame-rule seated --lock-elongated-polar --fov-resolution-mm "$FOV_MM" --chunk-points "$CHUNK"
        --roi-shape sphere --roi-radius-mm 100 --roi-zmin-mm -100 --train-boundary-points 3000
        --num-epochs "$EPOCHS" --lr "$ALR" --polar-lr "$TLR"
        --alpha-auto "$KAPPA" --alpha-max 1e6 --rms-weight 0.2
        --band-weight 1e7 --band-lower-t "$FIELD_T" --band-upper-t "$FIELD_T"
        --evaluate-every "$EVERY" --evaluation-points "$EVAL"
        --best-design "$OUT/stage${N}_best.pt" --log-each-n-epochs 1000 --wandb-mode disabled)
  if [ "$XY" != "-" ]; then
    ARGS+=(--positions --position-bound-mm "$XY" --position-bound-axial-mm "$Z"
           --position-base-design "$SEATS" --position-lr "$CLR" --refresh-every 100000000)
  fi
  say "stage $N: $EPOCHS epochs, angle lr $ALR, tilt lr $TLR, centre lr $CLR$([ "$XY" != "-" ] && echo ", box $XY/$Z mm around the seats")"
  if [ "$DRY" = 1 ]; then echo "$PYTHON -u scripts/run_dome_optimisation.py ${ARGS[*]}"; INIT="$OUT/stage${N}_best.pt"; continue; fi
  $PYTHON -u scripts/run_dome_optimisation.py "${ARGS[@]}" > "$OUT/stage${N}.log" 2>&1
  rc=$?
  [ -f "$OUT/stage${N}_best.pt" ] || { say "stage $N saved no design (rc $rc), stopping"; exit 1; }
  say "stage $N done (rc $rc), best $(best_of "$OUT/stage${N}.log") ppm"
  INIT="$OUT/stage${N}_best.pt"
done
[ "$DRY" = 1 ] && exit 0
cp "$INIT" "$OUT/final.pt"
say "final design $OUT/final.pt"
