"""
Paso 5: evaluacion final con incertidumbre (bootstrap sobre chips), brecha de generalizacion,
matriz de confusion y desglose por evento.

    python evaluate.py --groups gold_bce_dice weakft_bce_dice weakpre_bce_dice goldweak_bce_dice gold_bce gold_dice --ref gold_bce_dice

Para cada corrida runs/<grupo>_s<semilla>/ carga best.pt, predice valid/test/bolivia a 512x512 y guarda
los conteos por chip en eval_out/counts_*.csv (cache). El IoU de cada grupo es el promedio entre semillas;
el IC95% remuestrea chips (con reemplazo) y es PAREADO entre grupos (mismos chips en cada remuestreo).
"""
import argparse, json, os, pathlib
import numpy as np
import pandas as pd

SPLITS = ["valid", "test", "bolivia"]


def compute_counts(group, seed, data, out):
    f = out / f"counts_{group}_s{seed}.csv"
    if f.exists():
        return pd.read_csv(f)
    run = pathlib.Path(os.environ.get("RUNS_DIR", "runs")) / f"{group}_s{seed}"
    if not (run / "best.pt").exists():
        print(f"  [aviso] falta {run}/best.pt, se omite")
        return None
    import torch
    from torch.utils.data import DataLoader
    from data import FloodDataset, indices
    from unet import UNet
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    a = json.load(open(run / "metrics.json"))["args"]
    model = UNet(8 if a.get("s2") else 2, a["base"], a["depth"], skip=not a["no_skip"]).to(dev)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    model.eval()
    idx_df = pd.read_csv(pathlib.Path(data) / "gold_index.csv").set_index("i")
    rows = []
    print(f"  prediciendo {run.name} en {dev}...")
    for split in SPLITS:
        ids = indices(data, "gold", [split])
        sub = idx_df.loc[ids]
        k = 0
        with torch.no_grad():
            for x, y in DataLoader(FloodDataset(data, "gold", ids, extra="gold_S2.npy" if a.get("s2") else None), batch_size=4, shuffle=False):
                x, y = x.to(dev), y.to(dev)
                pred = model(x)[:, 0] > 0
                valid, t = y >= 0, y == 1
                c = [(pred & t & valid), (pred & ~t & valid), (~pred & t & valid), (~pred & ~t & valid)]
                c = [v.sum((1, 2)).cpu().numpy() for v in c]
                for j in range(len(x)):
                    r = sub.iloc[k]; k += 1
                    rows.append(dict(group=group, seed=seed, split=split, chip=r.chip, event=r.event,
                                     tp=int(c[0][j]), fp=int(c[1][j]), fn=int(c[2][j]), tn=int(c[3][j])))
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_processed")
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ref", default=None, help="grupo de referencia (por defecto el primero)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--subset", default="", help="archivo con nombres de chips: evalua solo sobre ellos (resultados en <eval>/subset/)")
    a = ap.parse_args()
    out = pathlib.Path(os.environ.get("EVAL_DIR", "eval_out")); out.mkdir(exist_ok=True)
    ref = a.ref or a.groups[0]

    frames = []
    for g in a.groups:
        for s in a.seeds:
            d = compute_counts(g, s, a.data, out)
            if d is not None:
                frames.append(d)
    D = pd.concat(frames)
    res = out
    if a.subset:
        keep = set(open(a.subset).read().split())
        D = D[D.chip.isin(keep)]
        res = out / "subset"; res.mkdir(exist_ok=True)
        print(f"Subconjunto: {len(keep)} chips en la lista; con datos: " +
              ", ".join(f"{s}={D[D.split == s].chip.nunique()}" for s in SPLITS))
    SPL = [s for s in SPLITS if (D.split == s).any()]
    groups = [g for g in a.groups if g in set(D.group)]
    assert ref in groups, f"no hay corridas del grupo de referencia {ref}"

    def arr(g, split, col):  # (semillas, chips), columnas ordenadas por nombre de chip
        return D[(D.group == g) & (D.split == split)].pivot(index="seed", columns="chip", values=col).values.astype(float)

    rng = np.random.default_rng(0)
    n = {s: arr(ref, s, "tp").shape[1] for s in SPL}
    IDX = {s: rng.integers(0, n[s], (a.boot, n[s])) for s in SPL}
    FULL = {s: np.arange(n[s])[None, :] for s in SPL}
    HAS_GAP = "test" in SPL and "bolivia" in SPL

    def metric(g, split, kind, idx):
        tp, fp, fn = (arr(g, split, c)[:, idx] for c in ("tp", "fp", "fn"))  # (S,B,N)
        if kind == "pooled":
            v = tp.sum(-1) / np.maximum((tp + fp + fn).sum(-1), 1)
        else:  # IoU medio por chip (chips sin agua en pred ni verdad se excluyen)
            u = tp + fp + fn
            with np.errstate(all="ignore"):
                v = np.nanmean(np.where(u > 0, tp / np.maximum(u, 1), np.nan), -1)
        return v.mean(0)

    def fmt(point, boots):
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"

    rows, grows, drows = [], [], []
    for g in groups:
        r = dict(grupo=g)
        for kind, nm in [("pooled", "IoU"), ("chip", "mIoU/chip")]:
            for s in SPL:
                r[f"{s} {nm}"] = fmt(metric(g, s, kind, FULL[s])[0], metric(g, s, kind, IDX[s]))
            if HAS_GAP:
                pt = metric(g, "test", kind, FULL["test"])[0] - metric(g, "bolivia", kind, FULL["bolivia"])[0]
                bt = metric(g, "test", kind, IDX["test"]) - metric(g, "bolivia", kind, IDX["bolivia"])
                r[f"brecha test-bolivia {nm}"] = fmt(pt, bt)
        rows.append(r)
        if g != ref:
            d = dict(grupo=f"{g} - {ref}")
            for s in SPL:
                pt = metric(g, s, "pooled", FULL[s])[0] - metric(ref, s, "pooled", FULL[s])[0]
                bt = metric(g, s, "pooled", IDX[s]) - metric(ref, s, "pooled", IDX[s])
                d[f"{s} dIoU"] = fmt(pt, bt)
            drows.append(d)
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 40)
    T = pd.DataFrame(rows).set_index("grupo")
    print(f"\n=== IoU agrupado, IC95% por bootstrap sobre chips (promedio de semillas {a.seeds}) ===")
    print(T[[f"{s} IoU" for s in SPL] + (["brecha test-bolivia IoU"] if HAS_GAP else [])].to_string())
    print("\n=== IoU medio por chip ===")
    print(T[[f"{s} mIoU/chip" for s in SPL] + (["brecha test-bolivia mIoU/chip"] if HAS_GAP else [])].to_string())
    T.to_csv(res / "resumen_bootstrap.csv")
    if drows:
        P = pd.DataFrame(drows).set_index("grupo")
        print(f"\n=== Diferencia de IoU frente a {ref} (pareada; si el IC incluye 0, no hay diferencia clara) ===")
        print(P.to_string())
        P.to_csv(res / "diferencias_vs_ref.csv")

    # ---- matriz de confusion (referencia, promedio de semillas)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CS = [s for s in ["test", "bolivia"] if s in SPL]
    fig, ax = plt.subplots(1, len(CS), figsize=(5 * len(CS), 4), squeeze=False)
    R = D[D.group == ref]
    print(f"\n=== Matriz de confusion de {ref} (filas = verdad, % de la fila) ===")
    for axi, s in zip(ax[0], CS):
        c = R[R.split == s].groupby("seed")[["tp", "fp", "fn", "tn"]].sum().mean()
        M = np.array([[c.tn, c.fp], [c.fn, c.tp]])
        P = M / M.sum(1, keepdims=True) * 100
        axi.imshow(P, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                axi.text(j, i, f"{P[i, j]:.1f}%\n({M[i, j] / 1e6:.2f} M px)", ha="center", va="center",
                         color="white" if P[i, j] > 50 else "black", fontsize=9)
        axi.set_xticks([0, 1], ["no agua", "agua"]); axi.set_yticks([0, 1], ["no agua", "agua"])
        axi.set_xlabel("prediccion"); axi.set_ylabel("verdad"); axi.set_title(s)
        print(f"{s}: no agua -> {P[0,0]:.1f}% ok, {P[0,1]:.1f}% falsa alarma | agua -> {P[1,1]:.1f}% detectada, {P[1,0]:.1f}% omitida")
    plt.tight_layout(); plt.savefig(res / "fig_confusion.png", dpi=130); plt.close()

    # ---- desglose por evento (referencia)
    G = R.groupby(["seed", "event"])[["tp", "fp", "fn", "tn"]].sum().reset_index()
    G["IoU"] = G.tp / (G.tp + G.fp + G.fn).clip(lower=1)
    G["precision"] = G.tp / (G.tp + G.fp).clip(lower=1)
    G["recall"] = G.tp / (G.tp + G.fn).clip(lower=1)
    G["agua_%"] = 100 * (G.tp + G.fn) / (G.tp + G.fp + G.fn + G.tn)
    E = G.groupby("event")[["IoU", "precision", "recall", "agua_%"]].mean()
    E["chips"] = R[R.seed == R.seed.min()].groupby("event").chip.nunique()
    print(f"\n=== Por evento ({ref}; chips valid+test+bolivia) ===")
    print(E.round(3).sort_values("IoU").to_string())
    E.to_csv(res / "por_evento.csv")
    print(f"\nGuardado en {res}/: resumen_bootstrap.csv, diferencias_vs_ref.csv, por_evento.csv, fig_confusion.png")


if __name__ == "__main__":
    main()
