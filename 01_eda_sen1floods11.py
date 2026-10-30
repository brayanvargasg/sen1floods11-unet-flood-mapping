"""
Paso 1: EDA de Sen1Floods11 (v1.1)

Uso (desde cualquier carpeta):
    python 01_eda_sen1floods11.py "/ruta/a/Sen1Floods11"

Genera la carpeta ./eda_out con tablas (CSV) y figuras (PNG).
Dependencias: pip install rasterio numpy pandas matplotlib tqdm
"""
import sys, re, pathlib, random
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser()
V = ROOT / "v1.1" if (ROOT / "v1.1").exists() else ROOT
HAND = V / "data/flood_events/HandLabeled"
WEAK = V / "data/flood_events/WeaklyLabeled"
OUT = pathlib.Path("eda_out")
OUT.mkdir(exist_ok=True)
random.seed(0)
np.random.seed(0)


def read(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def chip_id(name):
    return re.sub(r"_(S1Hand|LabelHand|S2Hand|JRCWaterHand|S1OtsuLabelHand|"
                  r"S1Weak|S2Weak|S1OtsuLabelWeak|S2IndexLabelWeak)(\.tif)?$", "", name)


# ---------------------------------------------------------------- 1) Particiones oficiales
print("\n=== 1) PARTICIONES OFICIALES ===")
split_of = {}
for csv in sorted((V / "splits").rglob("*.csv")):
    df = pd.read_csv(csv, header=None)
    ids = [chip_id(pathlib.Path(x).name) for x in df.iloc[:, 0]]
    print(f"{csv.relative_to(V)}: {len(df)} filas | ejemplo: {df.iloc[0].tolist()}")
    if "handlabeled" in str(csv).lower():
        tag = csv.stem.replace("flood_", "").replace("_data", "")
        for i in ids:
            split_of[i] = tag

# ---------------------------------------------------------------- 2) Chips con etiqueta dorada
print("\n=== 2) CHIPS DORADOS (HandLabeled) ===")
rows, vv_w, vv_l, vh_w, vh_l = [], [], [], [], []
label_files = sorted((HAND / "LabelHand").glob("*.tif"))
for lf in tqdm(label_files):
    cid = chip_id(lf.name)
    lab = read(lf)[0]
    s1 = read(HAND / "S1Hand" / f"{cid}_S1Hand.tif")
    otsu_p = HAND / "S1OtsuLabelHand" / f"{cid}_S1OtsuLabelHand.tif"
    otsu = read(otsu_p)[0] if otsu_p.exists() else None

    valid = lab != -1
    water, land = (lab == 1), (lab == 0)
    r = dict(chip=cid, event=cid.split("_")[0], split=split_of.get(cid, "sin_split"),
             h=lab.shape[0], w=lab.shape[1], bands_s1=s1.shape[0],
             water_frac=water.sum() / max(valid.sum(), 1),
             nodata_frac=(~valid).mean(),
             s1_nan_frac=np.isnan(s1).mean())
    for b, n in enumerate(["VV", "VH"]):
        r[f"{n}_mean"] = np.nanmean(s1[b]); r[f"{n}_min"] = np.nanmin(s1[b]); r[f"{n}_max"] = np.nanmax(s1[b])
    if otsu is not None:  # ruido de la etiqueta debil vs dorada (mismo chip)
        ov = (otsu == 1) & valid
        inter = (ov & water).sum(); union = ((otsu == 1) & valid | water).sum()
        r["otsu_vs_gold_IoU"] = inter / union if union else np.nan
    rows.append(r)

    # muestra de pixeles para histogramas VV/VH (agua vs tierra)
    ok = valid & ~np.isnan(s1).any(0)
    for mask, a_vv, a_vh in [(water & ok, vv_w, vh_w), (land & ok, vv_l, vh_l)]:
        idx = np.flatnonzero(mask)
        if len(idx) > 2000:
            idx = np.random.choice(idx, 2000, replace=False)
        a_vv.append(s1[0].ravel()[idx]); a_vh.append(s1[1].ravel()[idx])

df = pd.DataFrame(rows)
df.to_csv(OUT / "chips_dorados.csv", index=False)

summ_split = df.groupby("split").agg(chips=("chip", "count"), water_frac=("water_frac", "mean"),
                                     nodata_frac=("nodata_frac", "mean"),
                                     otsu_IoU=("otsu_vs_gold_IoU", "mean"))
summ_event = df.groupby("event").agg(chips=("chip", "count"), water_frac=("water_frac", "mean"),
                                     nodata_frac=("nodata_frac", "mean"),
                                     otsu_IoU=("otsu_vs_gold_IoU", "mean"))
print("\nPor particion:\n", summ_split.round(3))
print("\nPor evento:\n", summ_event.round(3))
summ_split.to_csv(OUT / "resumen_por_split.csv"); summ_event.to_csv(OUT / "resumen_por_evento.csv")
print("\nTamano de chips:", df[["h", "w"]].drop_duplicates().values.tolist(),
      "| bandas S1:", df.bands_s1.unique().tolist())
print("Chips con NaN en S1:", int((df.s1_nan_frac > 0).sum()))
print("Rango VV:", df.VV_min.min().round(1), "a", df.VV_max.max().round(1),
      "| VH:", df.VH_min.min().round(1), "a", df.VH_max.max().round(1))
print("Fraccion global de agua (sobre pixeles validos):", round(df.water_frac.mean(), 4))

# ---------------------------------------------------------------- 3) Chips con etiqueta debil
print("\n=== 3) CHIPS DEBILES (WeaklyLabeled) ===")
weak_files = sorted((WEAK / "S1OtsuLabelWeak").glob("*.tif"))
ev = pd.Series([chip_id(f.name).split("_")[0] for f in weak_files]).value_counts()
print(ev.to_string())
ev.to_csv(OUT / "debiles_por_evento.csv")
sample = random.sample(weak_files, min(300, len(weak_files)))
wf = []
for f in tqdm(sample):
    lab = read(f)[0]; v = lab != -1
    wf.append(((lab == 1) & v).sum() / max(v.sum(), 1))
print("Fraccion de agua en etiquetas debiles (muestra de 300):", round(float(np.mean(wf)), 4))

# ---------------------------------------------------------------- 4) Figuras
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
summ_event.water_frac.sort_values().plot.barh(ax=ax[0], title="Fraccion de agua por evento (dorada)")
ax[1].hist(df.water_frac, bins=30); ax[1].set_title("Fraccion de agua por chip"); ax[1].set_yscale("log")
ev.plot.barh(ax=ax[2], title="Chips debiles por evento")
plt.tight_layout(); plt.savefig(OUT / "fig_distribucion.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, (w, l, n) in zip(ax, [(vv_w, vv_l, "VV"), (vh_w, vh_l, "VH")]):
    a.hist(np.concatenate(l), bins=100, alpha=.5, density=True, label="no agua")
    a.hist(np.concatenate(w), bins=100, alpha=.5, density=True, label="agua")
    a.set_title(f"{n} (dB)"); a.legend()
plt.tight_layout(); plt.savefig(OUT / "fig_hist_vv_vh.png", dpi=130); plt.close()

# chips de ejemplo (uno por evento): VV, VH, etiqueta dorada, Otsu
evs = sorted(df.event.unique())
fig, ax = plt.subplots(len(evs), 4, figsize=(11, 2.6 * len(evs)))
for i, e in enumerate(evs):
    cid = df[df.event == e].sort_values("water_frac").iloc[-1].chip
    s1 = read(HAND / "S1Hand" / f"{cid}_S1Hand.tif")
    g = read(HAND / "LabelHand" / f"{cid}_LabelHand.tif")[0]
    o = read(HAND / "S1OtsuLabelHand" / f"{cid}_S1OtsuLabelHand.tif")[0]
    for j, (im, t, cm) in enumerate([(s1[0], "VV", "gray"), (s1[1], "VH", "gray"),
                                     (g, "Dorada", "viridis"), (o, "Otsu (debil)", "viridis")]):
        ax[i, j].imshow(im, cmap=cm); ax[i, j].axis("off")
        ax[i, j].set_title(f"{cid} | {t}" if j == 0 else t, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_ejemplos.png", dpi=110); plt.close()
print("\nListo. Resultados en:", OUT.resolve())
