#!/usr/bin/env bash
# Reproduce de extremo a extremo la Pregunta 2 (U-Net sobre Sen1Floods11).
#
#   bash run_all.sh /ruta/a/Sen1Floods11            # completo (3 semillas; ~5-6 h en un MacBook M1 Max)
#   BONUS_S2=1 bash run_all.sh /ruta/a/Sen1Floods11  # ademas ejecuta el bono SAR + optico (entregable 6)
#   QUICK=1 bash run_all.sh /ruta/a/Sen1Floods11    # verificacion rapida (1 semilla, pocas epocas; ~20 min)
#
# QUICK escribe en runs_quick/ y eval_out_quick/, sin tocar los resultados completos.
set -euo pipefail
ROOT=${1:?Uso: bash run_all.sh /ruta/a/Sen1Floods11}
export SEN1FLOODS11_ROOT="$ROOT"
export PYTORCH_ENABLE_MPS_FALLBACK=1
DATA=data_processed

if [ "${QUICK:-0}" = "1" ]; then
  export SEEDS="0" EPOCHS=2 EPOCHS_WEAK=1 EPOCHS_FT=2
  export RUNS_DIR=runs_quick EVAL_DIR=eval_out_quick SUMMARY_CSV=runs_quick_summary.csv
  echo ">>> MODO QUICK: resultados de verificacion, no usar en el informe"
else
  export SEEDS=${SEEDS:-"0 1 2"}
fi

echo "[1/8] EDA";                    python 01_eda_sen1floods11.py "$ROOT"
echo "[2/8] Preprocesamiento"
if [ -f "$DATA/stats.json" ]; then echo "   (ya existe $DATA/, se omite)"; else python 02_preprocess_sen1floods11.py "$ROOT" "$DATA"; fi

echo "[3/8] Ablacion de perdidas (solo dorada)";  bash run_ablation.sh loss
BEST=$(python - << 'PY'
import glob, json, os, re, collections
d = collections.defaultdict(list)
for f in glob.glob(os.environ.get("RUNS_DIR", "runs") + "/gold_*_s*/metrics.json"):
    m = re.search(r"/gold_(bce_dice|bce|dice)_s\d+/", f)
    if m:
        d[m.group(1)].append(json.load(open(f))["valid"]["IoU"])
print(max(d, key=lambda k: sum(d[k]) / len(d[k])))
PY
)
echo "   perdida elegida con 'valid': $BEST"

echo "[4/8] Ablacion de datos: dorada + debil";   bash run_ablation.sh data "$BEST"
echo "[5/8] Preentrenar en debil y afinar";       bash run_ablation.sh combine "$BEST"
echo "[6/8] Resumen y evaluacion con bootstrap"
python summarize.py
python evaluate.py --groups "gold_$BEST" "weakft_$BEST" "weakpre_$BEST" "goldweak_$BEST" gold_bce gold_dice \
       --ref "gold_$BEST" --seeds $SEEDS
echo "[7/8] Analisis cualitativo";                python qualitative.py --run "gold_${BEST}_s0"
echo "[8/8] Retrodispersion de los errores";      python backscatter_analysis.py --run "gold_${BEST}_s0"
if [ "${BONUS_S2:-0}" = "1" ]; then echo "[bono] Fusion SAR + Sentinel-2"; bash run_bonus_s2.sh "$ROOT"; fi
echo "Listo. Resultados en ${RUNS_DIR:-runs}/, ${EVAL_DIR:-eval_out}/ y ${SUMMARY_CSV:-runs_summary.csv}"
