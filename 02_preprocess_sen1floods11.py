"""
Paso 2: preprocesamiento de Sen1Floods11 (solo Sentinel-1 VV+VH y etiquetas).

Uso:
    python 02_preprocess_sen1floods11.py "/ruta/a/Sen1Floods11" [carpeta_salida]

Salida (por defecto ./data_processed):
    gold_X.npy  (N,2,512,512) float16  VV,VH normalizados
    gold_Y.npy  (N,512,512)   int8     -1 sin datos | 0 no agua | 1 agua
    gold_index.csv   i, chip, event, split(train/valid/test/bolivia)
    weak_X.npy / weak_Y.npy / weak_index.csv   (etiqueta debil: S1OtsuLabelWeak)
    stats.json  percentiles de recorte, media y desviacion (calculados SOLO con gold-train)
"""
import os, sys, re, json, pathlib
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

_root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SEN1FLOODS11_ROOT")
if not _root:
    sys.exit("Indica la ruta del dataset: python 02_preprocess_sen1floods11.py /ruta/a/Sen1Floods11 [salida]  "
             "(o define la variable de entorno SEN1FLOODS11_ROOT)")
ROOT = pathlib.Path(_root).expanduser()
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data_processed")
OUT.mkdir(exist_ok=True)
V = ROOT / "v1.1" if (ROOT / "v1.1").exists() else ROOT
HAND = V / "data/flood_events/HandLabeled"
WEAK = V / "data/flood_events/WeaklyLabeled"
SPL = V / "splits/flood_handlabeled"
H = W = 512
rng = np.random.default_rng(0)


def chip_id(name):
    return re.sub(r"_(S1Hand|LabelHand|S1Weak|S1OtsuLabelWeak)(\.tif)?$", "", pathlib.Path(name).name)


def read(path):
    with rasterio.open(path) as s:
        return s.read().astype(np.float32)


def fit(a, fill):
    """Fuerza forma (C,512,512) recortando o rellenando (por si algun chip difiere)."""
    out = np.full((a.shape[0], H, W), fill, np.float32)
    h, w = min(H, a.shape[1]), min(W, a.shape[2])
    out[:, :h, :w] = a[:, :h, :w]
    return out


def load(s1_path, lab_path):
    x = fit(read(s1_path)[:2], np.nan)
    y = fit(read(lab_path)[:1], -1)[0]
    y = np.nan_to_num(y, nan=-1)
    y[~np.isin(y, [0, 1])] = -1
    y[np.isnan(x).any(0)] = -1  # sin dato SAR no se puede predecir: se ignora en perdida y metricas
    return x, y.astype(np.int8)


# ---- tablas de chips ---------------------------------------------------------
gold = []
for split, f in [("train", "flood_train_data.csv"), ("valid", "flood_valid_data.csv"),
                 ("test", "flood_test_data.csv"), ("bolivia", "flood_bolivia_data.csv")]:
    d = pd.read_csv(SPL / f, header=None)
    for s1, lab in zip(d[0], d[1]):
        c = chip_id(s1)
        gold.append(dict(chip=c, event=c.split("_")[0], split=split,
                         s1=HAND / "S1Hand" / f"{c}_S1Hand.tif",
                         lab=HAND / "LabelHand" / f"{c}_LabelHand.tif"))
gold = pd.DataFrame(gold)

weak = []
for p in sorted((WEAK / "S1OtsuLabelWeak").glob("*.tif")):
    c = chip_id(p.name)
    weak.append(dict(chip=c, event=c.split("_")[0], split="weak",
                     s1=WEAK / "S1Weak" / f"{c}_S1Weak.tif", lab=p))
weak = pd.DataFrame(weak)

overlap = set(gold.chip) & set(weak.chip)
print(f"Gold: {len(gold)} | Weak: {len(weak)} | chips en comun gold/weak: {len(overlap)}")
assert not overlap, "Hay chips en ambos conjuntos: riesgo de fuga de datos"

# ---- estadisticas de normalizacion (solo gold-train) -------------------------
print("\nCalculando estadisticas con gold-train...")
samples = [[], []]
for r in tqdm(gold[gold.split == "train"].itertuples()):
    x = fit(read(r.s1)[:2], np.nan)
    for b in range(2):
        v = x[b][~np.isnan(x[b])]
        if len(v):
            samples[b].append(rng.choice(v, min(5000, len(v)), replace=False))
stats = {}
for b, n in enumerate(["VV", "VH"]):
    v = np.concatenate(samples[b])
    lo, hi = np.percentile(v, [0.5, 99.5])
    vc = np.clip(v, lo, hi)
    stats[n] = dict(lo=float(lo), hi=float(hi), mean=float(vc.mean()), std=float(vc.std()))
    print(f"{n}: recorte [{lo:.2f}, {hi:.2f}] dB | media {vc.mean():.2f} | std {vc.std():.2f}")
json.dump(stats, open(OUT / "stats.json", "w"), indent=2)


def normalize(x):
    out = np.empty_like(x)
    for b, n in enumerate(["VV", "VH"]):
        s = stats[n]
        out[b] = (np.clip(x[b], s["lo"], s["hi"]) - s["mean"]) / s["std"]
    return np.nan_to_num(out, nan=0.0)  # sin dato SAR -> 0 (= media)


def build(df, name):
    N = len(df)
    X = np.lib.format.open_memmap(OUT / f"{name}_X.npy", "w+", np.float16, (N, 2, H, W))
    Y = np.lib.format.open_memmap(OUT / f"{name}_Y.npy", "w+", np.int8, (N, H, W))
    nan_frac = []
    print(f"\nEscribiendo {name} ({N} chips)...")
    for i, r in enumerate(tqdm(df.itertuples(), total=N)):
        x, y = load(r.s1, r.lab)
        nan_frac.append(float(np.isnan(x).mean()))
        X[i], Y[i] = normalize(x).astype(np.float16), y
    X.flush(); Y.flush()
    idx = df[["chip", "event", "split"]].copy()
    idx.insert(0, "i", range(N)); idx["s1_nan_frac"] = nan_frac
    idx.to_csv(OUT / f"{name}_index.csv", index=False)
    v = Y[:]
    print(f"  X: min {float(X[:].min()):.2f} max {float(X[:].max()):.2f} | "
          f"Y: sin dato {np.mean(v == -1):.3f} no agua {np.mean(v == 0):.3f} agua {np.mean(v == 1):.3f}")
    print(f"  chips con NaN en S1: {int((np.array(nan_frac) > 0).sum())}")


build(gold, "gold")
build(weak, "weak")
print("\nListo ->", OUT.resolve())
