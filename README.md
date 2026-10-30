# Mapeo de inundaciones con Sentinel-1 (U-Net) — Pregunta 2

Redes Neuronales y Aprendizaje Profundo, UNI FIIS 2026-II (examen parcial). Segmentación píxel a píxel del agua de
inundación a partir de radar Sentinel-1 (bandas VV y VH) con una **U-Net implementada desde cero en PyTorch**, sobre el
dataset **Sen1Floods11** (Bonafilia et al., CVPR Workshops 2020).

## Ejecución de extremo a extremo (un comando)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh /ruta/a/Sen1Floods11          # completo: ~5-6 h en un MacBook M1 Max (GPU 'mps')
```

Verificación rápida (1 semilla, pocas épocas, ~20 min; escribe en `runs_quick/` y `eval_out_quick/`):

```bash
QUICK=1 bash run_all.sh /ruta/a/Sen1Floods11
```

`/ruta/a/Sen1Floods11` es la carpeta que contiene `v1.1/` (descarga del dataset: https://github.com/cloudtostreet/Sen1Floods11).
Detecta automáticamente `cuda`, `mps` o `cpu`.

## Estructura

| Archivo | Función |
|---|---|
| `01_eda_sen1floods11.py` | EDA: particiones oficiales, agua por evento, ruido de la etiqueta débil (Otsu vs dorada) |
| `02_preprocess_sen1floods11.py` | Solo S1 (VV, VH) + etiquetas → `data_processed/*.npy`. Normalización calculada solo con gold-train |
| `unet.py` | U-Net propia (profundidad, canales y *skip connections* configurables) |
| `data.py`, `losses.py` | Dataset con recortes 256×256 y aumento; pérdidas BCE/Dice/BCE+Dice y métricas, ignorando píxeles sin dato |
| `train.py` | Entrenamiento y evaluación (mejor época elegida con `valid`; `test` y Bolivia solo al final) |
| `run_ablation.sh`, `summarize.py` | Ablaciones (`loss`, `data`, `combine`, `arch`) con 3 semillas; tabla media ± desviación |
| `evaluate.py` | IC95% por bootstrap sobre chips, brecha test−Bolivia, diferencias pareadas, matriz de confusión, desglose por evento |
| `qualitative.py`, `backscatter_analysis.py` | Figuras de fallos y análisis de retrodispersión de los errores |
| `03_preprocess_s2.py`, `run_bonus_s2.sh` | Bono: 6 bandas Sentinel-2, detección de chips libres de nubes y comparación SAR vs SAR+óptico |
| `run_all.sh` | Orquesta todo lo anterior (`BONUS_S2=1` añade el bono) |

## Decisiones de diseño

- **Entrada:** VV y VH (dB), recortadas a los percentiles 0.5–99.5 y estandarizadas con estadísticas de gold-train. Los píxeles sin dato de radar se rellenan con 0 y **su etiqueta pasa a -1**.
- **Etiquetas:** -1 (sin dato) se ignora en pérdida y métricas. Los **chips débiles de Bolivia (224) nunca se usan** en entrenamiento.
- **Particiones:** las oficiales (252 train / 89 valid / 90 test / 15 Bolivia). Ni train, valid ni test contienen Bolivia, así que los modelos ya son "entrenados sin Bolivia".
- **Métricas:** IoU agrupado (sobre píxeles) e IoU medio por chip; el IC95% remuestrea chips y es pareado entre modelos.
- **Semillas:** 3 por configuración. La aleatoriedad en GPU (mps/cuda) puede producir diferencias pequeñas entre máquinas.

## Resultados principales (IoU agrupado, media ± desv. sobre 3 semillas)

Ablación de pérdida (solo dorada):

| Pérdida | valid | test | Bolivia |
|---|---|---|---|
| BCE | 0.631 ± 0.006 | 0.660 ± 0.004 | 0.678 ± 0.011 |
| Dice | 0.638 ± 0.005 | 0.667 ± 0.005 | 0.640 ± 0.020 |
| **BCE+Dice** | **0.647 ± 0.004** | **0.674 ± 0.003** | 0.645 ± 0.007 |

Uso de las etiquetas débiles (BCE+Dice, mismo presupuesto de pasos de gradiente):

| Estrategia | valid | test | Bolivia |
|---|---|---|---|
| Solo dorada | **0.647 ± 0.004** | **0.674 ± 0.003** | 0.645 ± 0.007 |
| Dorada + débil (mezcla) | 0.570 ± 0.009 | 0.615 ± 0.004 | 0.639 ± 0.037 |
| Preentrenar en débil → afinar en dorada | 0.633 ± 0.002 | 0.660 ± 0.004 | 0.630 ± 0.015 |
| Solo débil | 0.551 ± 0.005 | 0.600 ± 0.005 | 0.697 ± 0.016 |

Generalización a Bolivia (BCE+Dice): brecha test − Bolivia = 0.029, IC95% [−0.142, 0.254] → con 15 chips no hay evidencia de brecha.
Aun así, la omisión de agua sube de 20.0 % (test) a 28.5 % (Bolivia).

## Bono: fusión SAR + óptico (chips libres de nubes)

Subconjunto de 142 chips libres de nubes (76 train / 28 valid / 34 test / 4 Bolivia; sesgado hacia EE. UU., India y España). Mismo entrenamiento, 3 semillas, IoU agrupado:

| Modelo | valid | test |
|---|---|---|
| SAR + S2 (8 canales) | 0.890 | 0.902 |
| Solo SAR (mismo subconjunto) | 0.692 | 0.647 |
| Diferencia pareada (IC95%) | +0.199 [0.141, 0.302] | +0.254 [0.160, 0.375] |

Cautela: las etiquetas manuales probablemente provienen de imágenes ópticas, y el óptico no sirve bajo nubes. Con 4 chips de Bolivia no se concluye nada allí.

## Limitaciones

- Bolivia aporta solo 15 chips: las comparaciones en esa partición tienen intervalos muy anchos.
- Los diferentes eventos varían mucho (IoU de 0.31 en Pakistán a 0.86 en Mekong): un único país no representa "la generalización geográfica".
- Parte del error es de información (agua con retrodispersión similar a la tierra), no solo del modelo. Ver `eval_out/`.
- El bono (SAR + Sentinel-2) se ejecuta aparte con `bash run_bonus_s2.sh /ruta/a/Sen1Floods11`. Usa una heurística simple de nubes (brillo visible y banda de cirros), sin banda de calidad.

## Referencias

- Bonafilia, Tellman, Anderson, Issenberg. *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops, 2020.
- Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
