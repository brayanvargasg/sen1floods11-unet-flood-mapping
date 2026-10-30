"""Resume runs/*/metrics.json: media +- desviacion entre semillas por configuracion."""
import json, os, re, pathlib
import numpy as np
import pandas as pd

rows = []
for f in sorted(pathlib.Path(os.environ.get("RUNS_DIR", "runs")).glob("*/metrics.json")):
    name = f.parent.name
    m = re.match(r"(.+)_s(\d+)$", name)
    group, seed = (m.group(1), int(m.group(2))) if m else (name, 0)
    r = json.load(open(f))
    for split in ["valid", "test", "bolivia"]:
        for k in ["IoU", "mIoU_chip", "F1", "precision", "recall"]:
            rows.append(dict(group=group, seed=seed, split=split, metric=k, value=r[split][k]))
if not rows:
    raise SystemExit("No hay corridas en runs/*/metrics.json")
df = pd.DataFrame(rows)
n = df.groupby("group").seed.nunique()
agg = df.groupby(["group", "split", "metric"]).value.agg(["mean", "std"]).reset_index()
agg["txt"] = agg.apply(lambda r: f"{r['mean']:.3f}" + (f" ± {r['std']:.3f}" if not np.isnan(r["std"]) else ""), axis=1)
tab = agg.pivot_table(index="group", columns=["split", "metric"], values="txt", aggfunc="first")
tab.insert(0, "n_semillas", n)
tab.to_csv(os.environ.get("SUMMARY_CSV", "runs_summary.csv"))
pd.set_option("display.width", 250, "display.max_columns", 50)
for split in ["valid", "test", "bolivia"]:
    print(f"\n=== {split} ===")
    print(tab[split].join(n.rename("n")).to_string())
print("\nGuardado: runs_summary.csv")
