"""
Bono (entregable 6): prepara 6 bandas Sentinel-2 (B2,B3,B4,B8,B11,B12) de los chips manuales y detecta los chips libres de nubes.

    python 03_preprocess_s2.py /ruta/a/Sen1Floods11 [data_processed] [--max-cloud 0.02]

Requiere haber ejecutado antes 02_preprocess_sen1floods11.py. Genera en data_processed/:
    gold_S2.npy          (N,6,512,512) float16, mismo orden que gold_X.npy; estandarizado con estadisticas de gold-train
    s2_stats.json        recorte, media y desviacion por banda
    s2_chips.csv         por chip: fraccion de pixeles nublosos (heuristica), nodata de la etiqueta, indicador cloudfree
    cloudfree_chips.txt  nombres de los chips libres de nubes (los usan train.py --subset y evaluate.py --subset)
y eval_out/fig_s2_nubes.png para verificar visualmente el detector de nubes.

Deteccion de nubes (heuristica simple, sin banda de calidad): un pixel es 'nuboso' si el brillo visible medio (B2,B3,B4)
supera --bright, o si la banda de cirros B10 supera --cirrus. Ademas, un chip con nubes finas o bruma que cubre todo el
chip (no detectable pixel a pixel) se excluye si la mediana de B10 supera --b10-chip. Revisa eval_out/fig_s2_nubes.png y
eval_out/fig_s2_libres.png (chips 'libres' ordenados por B10) y ajusta los umbrales si hace falta.
"""
import os, sys, json, pathlib, argparse
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default=os.environ.get("SEN1FLOODS11_ROOT"))
ap.add_argument("out", nargs="?", default="data_processed")
ap.add_argument("--max-cloud", type=float, default=0.02, help="fraccion maxima de pixeles nublosos por chip")
ap.add_argument("--cirrus", type=float, default=0.01, help="umbral de reflectancia de la banda B10 (cirros)")
ap.add_argument("--bright", type=float, default=0.25, help="umbral de brillo visible medio")
ap.add_argument("--b10-chip", type=float, default=0.002,
                help="un chip se excluye si la mediana de B10 (cirros/bruma fina) supera este valor")
a = ap.parse_args()
if not a.root:
    sys.exit("Indica la ruta del dataset: python 03_preprocess_s2.py /ruta/a/Sen1Floods11 [data_processed]")

ROOT = pathlib.Path(a.root).expanduser()
V = ROOT / "v1.1" if (ROOT / "v1.1").exists() else ROOT
S2DIR = V / "data/flood_events/HandLabeled/S2Hand"
OUT = pathlib.Path(a.out)
H = W = 512
SEL = [1, 2, 3, 7, 11, 12]          # B2, B3, B4, B8, B11, B12 dentro de las 13 bandas del archivo
NAMES = ["B2", "B3", "B4", "B8", "B11", "B12"]
B10 = 10
rng = np.random.default_rng(0)

idx = pd.read_csv(OUT / "gold_index.csv")
Y = np.load(OUT / "gold_Y.npy", mmap_mode="r")
N = len(idx)


def read(chip):
    with rasterio.open(S2DIR / f"{chip}_S2Hand.tif") as s:
        x = s.read().astype(np.float32)
    out = np.full((13, H, W), np.nan, np.float32)
    c, h, w = min(13, x.shape[0]), min(H, x.shape[1]), min(W, x.shape[2])
    out[:c, :h, :w] = x[:c, :h, :w]
    return out


# ---- unidades: DN (0-10000) o reflectancia (0-1)
probe = [read(c) for c in idx.chip[:20]]
p99 = np.nanpercentile(np.stack([p[3] for p in probe]), 99)
scale = 1e4 if p99 > 2 else 1.0
print(f"B4: percentil 99 = {p99:.3f} -> escala de reflectancia: dividir por {scale:g}")
print("Mediana por banda (13) en 20 chips:", np.round(np.nanmedian(np.stack(probe), axis=(0, 2, 3)) / scale, 4).tolist())
del probe

# ---- pasada 1: reflectancias crudas + proxy de nubes
Xs = np.lib.format.open_memmap(OUT / "gold_S2.npy", "w+", np.float16, (N, len(SEL), H, W))
rows, samples = [], [[] for _ in SEL]
for r in tqdm(idx.itertuples(), total=N):
    refl = read(r.chip) / scale
    valid = ~np.isnan(refl).any(0) & ~(refl[SEL] == 0).all(0)   # NaN o relleno de ceros = sin dato S2
    vis = refl[[1, 2, 3]].mean(0)
    cloudy = ((vis > a.bright) | (refl[B10] > a.cirrus)) & valid
    x6 = np.clip(refl[SEL], 0, 2)
    x6[:, ~valid] = np.nan
    Xs[r.i] = x6.astype(np.float16)
    rows.append(dict(i=r.i, chip=r.chip, event=r.event, split=r.split, cloud_frac=float(cloudy.mean()),
                     b10_med=float(np.median(refl[B10][valid])) if valid.any() else np.nan,
                     s2_nan_frac=float((~valid).mean()), label_nodata_frac=float((np.array(Y[r.i]) == -1).mean())))
    if r.split == "train":
        for b in range(len(SEL)):
            v = refl[SEL[b]][valid]
            if len(v):
                samples[b].append(rng.choice(v, min(5000, len(v)), replace=False))
Xs.flush()
M = pd.DataFrame(rows)
M["cloudfree"] = (M.cloud_frac <= a.max_cloud) & (M.s2_nan_frac <= 0.01) & (M.b10_med <= a.b10_chip)
M.to_csv(OUT / "s2_chips.csv", index=False)
(OUT / "cloudfree_chips.txt").write_text("\n".join(M[M.cloudfree].chip) + "\n")

print(f"\nChips libres de nubes (cloud_frac <= {a.max_cloud}): {int(M.cloudfree.sum())} de {N}")
print(M.groupby("split").cloudfree.agg(["sum", "count"]).rename(columns={"sum": "libres", "count": "total"}).to_string())
print("\nPor evento:\n", M.groupby("event").cloudfree.agg(["sum", "count"]).rename(columns={"sum": "libres", "count": "total"}).to_string())
print("\nSensibilidad al umbral de nubes (chips libres):",
      {t: int(((M.cloud_frac <= t) & (M.s2_nan_frac <= 0.01)).sum()) for t in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]})
print("Cuantiles de cloud_frac:", M.cloud_frac.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist())
print("Cuantiles de la mediana de B10:", M.b10_med.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(4).tolist())
print("Chips con cloud_frac <= %.2f pero excluidos por B10 (> %.4f): %d" % (a.max_cloud, a.b10_chip, int(((M.cloud_frac <= a.max_cloud) & (M.b10_med > a.b10_chip)).sum())))

# ---- figura de verificacion del detector de nubes: chips repartidos por cloud_frac
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = M.sort_values("cloud_frac").reset_index(drop=True)
pick = order.iloc[np.linspace(0, N - 1, 12).astype(int)]
fig, ax = plt.subplots(3, 4, figsize=(12, 9))
for a_, r in zip(ax.ravel(), pick.itertuples()):
    rgb = np.array(Xs[r.i][[2, 1, 0]], dtype=np.float32).transpose(1, 2, 0)   # B4,B3,B2
    a_.imshow(np.clip(np.nan_to_num(rgb) / 0.3, 0, 1)); a_.set_xticks([]); a_.set_yticks([])
    a_.set_title(f"{r.chip}\nnubes {r.cloud_frac:.2f} | B10 {r.b10_med:.4f}", fontsize=8)
pathlib.Path("eval_out").mkdir(exist_ok=True)
plt.tight_layout(); plt.savefig("eval_out/fig_s2_nubes.png", dpi=90); plt.close()

# segunda figura: chips que pasarian solo el filtro de nubes por pixel, ordenados por B10 (para ver la bruma fina)
cand = M[(M.cloud_frac <= a.max_cloud) & (M.s2_nan_frac <= 0.01)].sort_values("b10_med").reset_index(drop=True)
if len(cand):
    pick2 = cand.iloc[np.linspace(0, len(cand) - 1, min(12, len(cand))).astype(int)]
    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for a_ in ax.ravel():
        a_.axis("off")
    for a_, r in zip(ax.ravel(), pick2.itertuples()):
        rgb = np.array(Xs[r.i][[2, 1, 0]], dtype=np.float32).transpose(1, 2, 0)
        a_.imshow(np.clip(np.nan_to_num(rgb) / 0.3, 0, 1)); a_.set_xticks([]); a_.set_yticks([]); a_.axis("on")
        a_.set_title(f"{r.chip}\nnubes {r.cloud_frac:.2f} | B10 {r.b10_med:.4f}" + (" | EXCLUIDO" if r.b10_med > a.b10_chip else ""), fontsize=8)
    plt.tight_layout(); plt.savefig("eval_out/fig_s2_libres.png", dpi=90); plt.close()

# ---- estadisticas (solo gold-train) y normalizacion en sitio
stats = {}
for b, n in enumerate(NAMES):
    v = np.concatenate(samples[b])
    lo, hi = np.percentile(v, [0.5, 99.5])
    vc = np.clip(v, lo, hi)
    stats[n] = dict(lo=float(lo), hi=float(hi), mean=float(vc.mean()), std=float(vc.std()))
    print(f"{n}: recorte [{lo:.4f}, {hi:.4f}] | media {vc.mean():.4f} | std {vc.std():.4f}")
json.dump(dict(scale=scale, bands=stats), open(OUT / "s2_stats.json", "w"), indent=2)
for i in tqdm(range(N), desc="normalizando"):
    x = np.array(Xs[i], dtype=np.float32)
    for b, n in enumerate(NAMES):
        s = stats[n]
        x[b] = (np.clip(x[b], s["lo"], s["hi"]) - s["mean"]) / s["std"]
    Xs[i] = np.nan_to_num(x, nan=0.0).astype(np.float16)
Xs.flush()
print("\nListo ->", OUT.resolve(), "(gold_S2.npy, s2_chips.csv, cloudfree_chips.txt) y eval_out/fig_s2_nubes.png / fig_s2_libres.png")
