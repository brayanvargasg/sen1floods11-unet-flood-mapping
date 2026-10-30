"""
Cuantifica los errores del modelo en terminos de retrodispersion (dB) para respaldar el analisis cualitativo.

    python backscatter_analysis.py --run gold_bce_dice_s0

Salida: eval_out/fig_retrodispersion.png y tablas impresas:
  (1) mediana de VV, VH y VV-VH (dB) de los pixeles TP / FN / FP / TN por particion
  (2) tasa de omision del agua segun el nivel de retrodispersion (si sube con dB, el modelo falla en agua 'no oscura')
"""
import argparse, json, os, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VV_BINS = [-60, -25, -20, -15, -10, -5, 20]
VH_BINS = [-60, -30, -25, -20, -15, -10, 20]


def lab(b, i):
    lo, hi = b[i], b[i + 1]
    return f"< {hi}" if i == 0 else (f">= {lo}" if i == len(b) - 2 else f"{lo} a {hi}")


def report(samples, hist, out):
    S = pd.concat(samples)
    S["vv_vh"] = S.vv - S.vh
    print("\n=== (1) Mediana de retrodispersion por categoria (dB) ===")
    print(S.groupby(["split", "cat"])[["vv", "vh", "vv_vh"]].median().round(1).unstack("cat").to_string())
    for sp in [s for s in ["valid", "test", "bolivia"] if s in hist]:
        for band, bins in [("vv", VV_BINS), ("vh", VH_BINS)]:
            a, f = hist[sp][band + "_all"], hist[sp][band + "_fn"]
            T = pd.DataFrame({f"{band.upper()} (dB)": [lab(bins, i) for i in range(len(a))],
                              "% del agua real": (100 * a / max(a.sum(), 1)).round(1),
                              "% omitido": (100 * f / np.maximum(a, 1)).round(1)})
            print(f"\n=== (2) {sp}: omision del agua segun {band.upper()} ===")
            print(T.to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    P = S[S.split.isin(["test", "bolivia"])]
    for a, col in zip(ax, ["vv", "vh"]):
        for c, color in [("TP", "tab:green"), ("FN", "tab:orange"), ("FP", "tab:red"), ("TN", "gray")]:
            a.hist(P[P.cat == c][col], bins=80, range=(-40, 5), density=True, histtype="step", color=color, label=c)
        a.set_xlabel(f"{col.upper()} (dB)"); a.set_ylabel("densidad"); a.legend()
    ax[0].set_title("test + Bolivia: retrodispersion por categoria")
    plt.tight_layout(); plt.savefig(out / "fig_retrodispersion.png", dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="gold_bce_dice_s0")
    ap.add_argument("--data", default="data_processed")
    ap.add_argument("--per-chip", type=int, default=3000)
    a = ap.parse_args()
    import torch
    from unet import UNet
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    run = pathlib.Path(os.environ.get("RUNS_DIR", "runs")) / a.run
    args = json.load(open(run / "metrics.json"))["args"]
    model = UNet(2, args["base"], args["depth"], skip=not args["no_skip"]).to(dev)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev)); model.eval()
    data = pathlib.Path(a.data)
    st = json.load(open(data / "stats.json"))
    idx = pd.read_csv(data / "gold_index.csv")
    idx = idx[idx.split.isin(["valid", "test", "bolivia"])].reset_index(drop=True)
    X = np.load(data / "gold_X.npy", mmap_mode="r"); Y = np.load(data / "gold_Y.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    samples = []
    hist = {s: {k: np.zeros(len(b) - 1) for k, b in [("vv_all", VV_BINS), ("vv_fn", VV_BINS),
                                                      ("vh_all", VH_BINS), ("vh_fn", VH_BINS)]}
            for s in ["valid", "test", "bolivia"]}
    print(f"Prediciendo {len(idx)} chips con {a.run} en {dev}...")
    with torch.no_grad():
        for s in range(0, len(idx), 4):
            sub = idx.iloc[s:s + 4]
            xb = np.stack([np.array(X[i], dtype=np.float32) for i in sub.i])
            pb = (model(torch.from_numpy(xb).to(dev))[:, 0] > 0).cpu().numpy()
            for k, (_, r) in enumerate(sub.iterrows()):
                y = np.array(Y[r.i], dtype=np.int64)
                p, v, t = pb[k], y >= 0, y == 1
                vv = (xb[k, 0] * st["VV"]["std"] + st["VV"]["mean"]).ravel()
                vh = (xb[k, 1] * st["VH"]["std"] + st["VH"]["mean"]).ravel()
                cats = {"TP": p & t & v, "FN": ~p & t & v, "FP": p & ~t & v, "TN": ~p & ~t & v}
                for c, m in cats.items():
                    ids = np.flatnonzero(m.ravel())
                    if len(ids) > a.per_chip:
                        ids = rng.choice(ids, a.per_chip, replace=False)
                    samples.append(pd.DataFrame(dict(split=r.split, cat=c, vv=vv[ids], vh=vh[ids])))
                w, fn = (t & v).ravel(), cats["FN"].ravel()
                for band, val, bins in [("vv", vv, VV_BINS), ("vh", vh, VH_BINS)]:
                    hist[r.split][band + "_all"] += np.histogram(val[w], bins)[0]
                    hist[r.split][band + "_fn"] += np.histogram(val[fn], bins)[0]
    out = pathlib.Path(os.environ.get("EVAL_DIR", "eval_out")); out.mkdir(exist_ok=True)
    report(samples, hist, out)
    print("\nGuardado: eval_out/fig_retrodispersion.png")


if __name__ == "__main__":
    main()
