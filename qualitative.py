"""
Paso 6: analisis cualitativo de fallos. Genera paginas de figuras con VV, VH, verdad, prediccion y mapa de error
para los chips mas informativos (peor IoU, mayor omision, mayor falsa alarma, peores de Bolivia, mejores).

    python qualitative.py --run gold_bce_dice_s0

Salida: eval_out/quali/quali_*.png y eval_out/quali/casos.csv
Mapa de error: verde = acierto (agua), rojo = falsa alarma, amarillo = agua omitida, gris = sin dato.
"""
import argparse, json, os, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, WATER, NODATA = 0.12, [0.1, 0.45, 1.0], [0.55, 0.55, 0.55]


def rgb_labels(y, water):
    img = np.full(y.shape + (3,), BG)
    img[water] = WATER
    img[y == -1] = NODATA
    return img


def rgb_error(y, pred):
    img = np.full(y.shape + (3,), BG)
    v, t = y >= 0, y == 1
    img[v & pred & t] = [0.1, 0.8, 0.2]
    img[v & pred & ~t] = [1.0, 0.1, 0.1]
    img[v & ~pred & t] = [1.0, 0.85, 0.1]
    img[y == -1] = NODATA
    return img


def page(items, fname, titulo):
    n = len(items)
    fig, ax = plt.subplots(n, 5, figsize=(15, 3.1 * n), squeeze=False)
    heads = ["VV", "VH", "Verdad (experto)", "Prediccion", "Error: verde acierto / rojo falsa alarma / amarillo omitida"]
    for r, it in enumerate(items):
        x, y, pred = it["x"].astype(np.float32), it["y"], it["pred"]
        panels = [(x[0], dict(cmap="gray", vmin=np.percentile(x[0], 1), vmax=np.percentile(x[0], 99))),
                  (x[1], dict(cmap="gray", vmin=np.percentile(x[1], 1), vmax=np.percentile(x[1], 99))),
                  (rgb_labels(y, y == 1), {}), (rgb_labels(y, pred), {}), (rgb_error(y, pred), {})]
        for c, (im, kw) in enumerate(panels):
            a = ax[r, c]
            a.imshow(im, **kw)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(heads[c], fontsize=8)
        ax[r, 0].set_ylabel(f"{it['chip']}\n{it['split']} | IoU {it['iou']:.2f}\nP {it['prec']:.2f} R {it['rec']:.2f}\n"
                            f"agua {it['water_pct']:.0f}%", fontsize=8)
    fig.suptitle(titulo, fontsize=11)
    plt.tight_layout()
    plt.savefig(fname, dpi=75)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="gold_bce_dice_s0")
    ap.add_argument("--data", default="data_processed")
    ap.add_argument("--n", type=int, default=4, help="chips por pagina")
    a = ap.parse_args()
    import torch
    from unet import UNet
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    run = pathlib.Path(os.environ.get("RUNS_DIR", "runs")) / a.run
    args = json.load(open(run / "metrics.json"))["args"]
    model = UNet(2, args["base"], args["depth"], skip=not args["no_skip"]).to(dev)
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    model.eval()

    data = pathlib.Path(a.data)
    idx = pd.read_csv(data / "gold_index.csv")
    idx = idx[idx.split.isin(["valid", "test", "bolivia"])].reset_index(drop=True)
    X = np.load(data / "gold_X.npy", mmap_mode="r")
    Y = np.load(data / "gold_Y.npy", mmap_mode="r")

    preds, rows = {}, []
    print(f"Prediciendo {len(idx)} chips con {a.run} en {dev}...")
    with torch.no_grad():
        for s in range(0, len(idx), 4):
            sub = idx.iloc[s:s + 4]
            x = torch.from_numpy(np.stack([np.array(X[i], dtype=np.float32) for i in sub.i])).to(dev)
            p = (model(x)[:, 0] > 0).cpu().numpy()
            for k, (_, r) in enumerate(sub.iterrows()):
                y = np.array(Y[r.i], dtype=np.int64)
                v, t = y >= 0, y == 1
                tp, fp = int((p[k] & t & v).sum()), int((p[k] & ~t & v).sum())
                fn = int((~p[k] & t & v).sum())
                u = tp + fp + fn
                preds[r.i] = p[k]
                rows.append(dict(i=r.i, chip=r.chip, event=r.event, split=r.split, tp=tp, fp=fp, fn=fn,
                                 iou=tp / u if u else np.nan, prec=tp / (tp + fp) if tp + fp else 0.0,
                                 rec=tp / (tp + fn) if tp + fn else 0.0,
                                 water_pct=100 * t.sum() / max(v.sum(), 1), nodata_pct=100 * (~v).mean()))
    M = pd.DataFrame(rows)
    Mw = M[M.water_pct >= 1]
    sel = {
        "peores_iou": (Mw.nsmallest(2 * a.n, "iou"), "Peor IoU (chips con >=1% de agua)"),
        "mayor_omision": (M.nlargest(a.n, "fn"), "Mayor agua omitida (pixeles FN)"),
        "mayor_falsa_alarma": (M.nlargest(a.n, "fp"), "Mayor falsa alarma (pixeles FP)"),
        "bolivia_peores": (Mw[Mw.split == "bolivia"].nsmallest(a.n, "iou"), "Bolivia: peores chips"),
        "mejores": (Mw[Mw.water_pct >= 5].nlargest(a.n, "iou"), "Mejores chips (referencia)"),
    }
    out = pathlib.Path(os.environ.get("EVAL_DIR", "eval_out")) / "quali"; out.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, (df, titulo) in sel.items():
        for pg in range(0, len(df), a.n):
            part = df.iloc[pg:pg + a.n]
            items = [dict(x=np.array(X[r.i]), y=np.array(Y[r.i], dtype=np.int64), pred=preds[r.i], **r._asdict())
                     for r in part.itertuples(index=False)]
            fn = out / f"quali_{name}{'' if len(df) <= a.n else '_p' + str(pg // a.n + 1)}.png"
            page(items, fn, f"{titulo} | modelo {a.run}")
            print("  ", fn)
        cases.append(df.assign(categoria=name))
    C = pd.concat(cases)
    C.to_csv(out / "casos.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", C[["categoria", "chip", "split", "iou", "prec", "rec", "water_pct", "nodata_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
