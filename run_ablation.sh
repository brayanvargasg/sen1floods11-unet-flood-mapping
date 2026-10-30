#!/usr/bin/env bash
# Ablaciones del examen (Pregunta 2). Ejecutar desde la carpeta con train.py, con el entorno activo.
#   bash run_ablation.sh loss               # (b) BCE vs Dice vs BCE+Dice, solo dorada      (9 corridas, ~2 h)
#   bash run_ablation.sh data bce_dice      # (a) dorada + debil con la mejor perdida        (3 corridas, ~40 min)
#   bash run_ablation.sh combine bce_dice   # preentrenar en debil y afinar en dorada       (6 corridas, ~50 min)
#   bash run_ablation.sh arch bce_dice      # arquitectura: sin skip, profundidad 3 y 5      (9 corridas, ~2 h)
# Variables opcionales: SEEDS="0 1 2"  DATA=data_processed  EPOCHS=30  EPOCHS_WEAK=6
# Si una corrida ya tiene runs/<nombre>/metrics.json se omite (puedes interrumpir y continuar).
set -e
DATA=${DATA:-data_processed}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-30}
EPOCHS_FT=${EPOCHS_FT:-15}
RUNS_DIR=${RUNS_DIR:-runs}
EPOCHS_WEAK=${EPOCHS_WEAK:-6}   # 6 epocas x ~646 pasos ~ 30 epocas x 126 pasos: mismo presupuesto de computo
export PYTORCH_ENABLE_MPS_FALLBACK=1

run() {
  name=$1; shift
  if [ -f "$RUNS_DIR/$name/metrics.json" ]; then echo ">> omito $name (ya existe)"; return; fi
  echo ">> $name"
  python train.py --data "$DATA" --name "$name" "$@"
}

case "$1" in
  loss)
    for l in bce dice bce_dice; do for s in $SEEDS; do
      run "gold_${l}_s$s" --sets gold --loss "$l" --epochs "$EPOCHS" --seed "$s"
    done; done ;;
  data)
    L=${2:?indica la perdida ganadora, p. ej.: bash run_ablation.sh data bce_dice}
    for s in $SEEDS; do
      run "goldweak_${L}_s$s" --sets gold+weak --loss "$L" --epochs "$EPOCHS_WEAK" --seed "$s"
    done ;;
  combine)
    L=${2:-bce_dice}
    for s in $SEEDS; do
      run "weakpre_${L}_s$s" --sets weak --loss "$L" --epochs "$EPOCHS_WEAK" --seed "$s"
      run "weakft_${L}_s$s" --sets gold --loss "$L" --epochs "$EPOCHS_FT" --lr 3e-4 --seed "$s" --init "$RUNS_DIR/weakpre_${L}_s$s/best.pt"
    done ;;
  arch)
    L=${2:-bce_dice}
    for s in $SEEDS; do
      run "arch_noskip_${L}_s$s" --sets gold --loss "$L" --epochs "$EPOCHS" --seed "$s" --no-skip
      run "arch_d3_${L}_s$s"     --sets gold --loss "$L" --epochs "$EPOCHS" --seed "$s" --depth 3
      run "arch_d5_${L}_s$s"     --sets gold --loss "$L" --epochs "$EPOCHS" --seed "$s" --depth 5
    done ;;
  *) echo "Uso: bash run_ablation.sh {loss|data <perdida>|combine [perdida]|arch [perdida]}"; exit 1 ;;
esac
echo "Listo. Resume con: python summarize.py"
