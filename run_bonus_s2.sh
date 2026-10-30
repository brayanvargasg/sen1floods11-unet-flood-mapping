#!/usr/bin/env bash
# Bono (entregable 6): fusion SAR + optico (Sentinel-2) en chips libres de nubes.
#   bash run_bonus_s2.sh /ruta/a/Sen1Floods11
# Entrena, con las mismas semillas y el mismo subconjunto de chips libres de nubes, un modelo solo-SAR y un modelo SAR+S2,
# y los compara con IC95% pareado sobre chips (junto al modelo solo-SAR entrenado con todos los chips).
# Variables opcionales: SEEDS="0 1 2" EPOCHS=30 LOSS=bce_dice DATA=data_processed RUNS_DIR=runs
set -euo pipefail
ROOT=${1:?Uso: bash run_bonus_s2.sh /ruta/a/Sen1Floods11}
export SEN1FLOODS11_ROOT="$ROOT" PYTORCH_ENABLE_MPS_FALLBACK=1
DATA=${DATA:-data_processed}; SEEDS=${SEEDS:-"0 1 2"}; EPOCHS=${EPOCHS:-30}; L=${LOSS:-bce_dice}; RUNS_DIR=${RUNS_DIR:-runs}
SUB="$DATA/cloudfree_chips.txt"

[ -f "$DATA/gold_S2.npy" ] || python 03_preprocess_s2.py "$ROOT" "$DATA"

# recortes por chip y epoca, para igualar ~1008 recortes/epoca del modelo principal aunque haya pocos chips
REP=$(python - << PY
import pandas as pd
d = pd.read_csv("$DATA/gold_index.csv"); keep = set(open("$SUB").read().split())
n = int(((d.split == "train") & d.chip.isin(keep)).sum())
print(max(4, round(1008 / max(n, 1))))
PY
)
echo "Chips de entrenamiento libres de nubes: repeticion por epoca = $REP"

for s in $SEEDS; do
  for v in sar fused; do
    name="s2${v}_${L}_s$s"; extra=""; [ "$v" = "fused" ] && extra="--s2"
    if [ -f "$RUNS_DIR/$name/metrics.json" ]; then echo ">> omito $name"; continue; fi
    echo ">> $name"
    python train.py --data "$DATA" --sets gold --loss "$L" --epochs "$EPOCHS" --seed "$s" \
        --subset "$SUB" --gold-repeat "$REP" $extra --name "$name"
  done
done
python evaluate.py --groups "s2fused_$L" "s2sar_$L" "gold_$L" --ref "s2sar_$L" --subset "$SUB" --seeds $SEEDS
