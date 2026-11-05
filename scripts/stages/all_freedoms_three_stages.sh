#!/usr/bin/env bash
# Every freedom open from the first iteration, in three stages with falling steps:
# magnets turn and tilt, centres move within 2 mm in their plate or ring plane per
# segment (the box re-centres every 1,500 epochs), a penalty keeps a 0.2 mm gap
# between bodies, and each stage restarts from the previous stage's best design.
#
#   OUT=runs/x PROBLEM="--config ... --evaluation-points ..." [SEED=design.pt] \
#   bash scripts/stages/all_freedoms_three_stages.sh
#
# PROBLEM holds the flags that describe the problem: geometry, region, field pin and
# evaluation set. Without SEED the optimizer builds and seeds the arrangement itself.
# Stage 1 may keep a best design with overlaps as a restart point. Stages 2 and 3 keep
# the strict overlap gate unless ALLOW_OVERLAPS=1. STAGE_EPOCHS="e1 e2 e3" shortens a
# test run, and DRY=1 prints the commands. The final design is $OUT/final.pt.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:?set OUT}"; PROBLEM="${PROBLEM:?set PROBLEM}"; SEED="${SEED:-}"
PYTHON="${PYTHON:-python3}"; DRY="${DRY:-0}"
read -r -a PROBLEM_ARGS <<< "$PROBLEM"
read -r E1 E2 E3 <<< "${STAGE_EPOCHS:-80000 60000 60000}"
mkdir -p "$OUT"; LOG="$OUT/stages.log"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }
best_of() { grep -F '[truth]' "$1" | awk '{print $4}' | sort -n | head -1; }
say "three stages with all freedoms in $OUT: seed ${SEED:-built and seeded by the optimizer}"

#        stage  epochs  lr      sharpness
LADDER=("1 $E1 0.002  10"
        "2 $E2 0.001  20"
        "3 $E3 0.0005 20")
INIT="$SEED"
for row in "${LADDER[@]}"; do
  read -r N EPOCHS LR SHARP <<< "$row"
  EVERY=$(( EPOCHS < 4000 ? EPOCHS / 2 : 2000 ))
  ARGS=("${PROBLEM_ARGS[@]}" --double-precision
        --positions --in-plane --position-lr 1e-5 --position-bound-mm 2 --refresh-every 1500
        --collision-weight 1e10 --collision-margin-mm=-0.2
        --num-epochs "$EPOCHS" --lr "$LR" --alpha-auto "$SHARP" --alpha-max 5e5 --rms-weight 0.2
        --evaluate-every "$EVERY"
        --best-design "$OUT/stage${N}_best.pt" --log-each-n-epochs 500 --wandb-mode disabled)
  [ -n "$INIT" ] && ARGS+=(--init-design "$INIT")
  { [ "$N" = 1 ] || [ "${ALLOW_OVERLAPS:-0}" = 1 ]; } && ARGS+=(--best-allow-overlaps)
  say "stage $N: $EPOCHS epochs, lr $LR, sharpness $SHARP${INIT:+, from $INIT}"
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
