"""
Entrenamiento y evaluacion de la U-Net.

Prueba rapida (humo):
    python train.py --data data_processed --sets gold --epochs 1 --max-steps 20 --name humo
Corrida normal:
    python train.py --data data_processed --sets gold --loss bce_dice --epochs 40 --name gold_bcedice
Preentrenar en debil y afinar en dorada (ver run_ablation.sh combine):
    python train.py --sets weak --epochs 6 --name pre ; python train.py --sets gold --init runs/pre/best.pt --lr 3e-4 --name ft
Con etiquetas debiles:
    python train.py --data data_processed --sets gold+weak --loss bce_dice --epochs 20 --name goldweak_bcedice

Reglas del diseno:
 * Entrenamiento: chips dorados 'train' (+ debiles SIN Bolivia si --sets gold+weak). Bolivia nunca se entrena.
 * La mejor epoca se elige con 'valid'. 'test' y 'bolivia' solo se miden al final.
"""
import argparse, json, os, pathlib, time
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from data import FloodDataset, indices
from losses import get_loss, evaluate
from unet import UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_processed")
    ap.add_argument("--sets", choices=["gold", "weak", "gold+weak"], default="gold")
    ap.add_argument("--init", default="", help="checkpoint .pt para inicializar (preentrenar en debil y afinar en dorada)")
    ap.add_argument("--s2", action="store_true", help="anade 6 bandas Sentinel-2 como canales de entrada (solo etiquetas manuales)")
    ap.add_argument("--subset", default="", help="archivo con nombres de chips; restringe entrenamiento y evaluacion a ellos")
    ap.add_argument("--loss", choices=["bce", "dice", "bce_dice"], default="bce_dice")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--gold-repeat", type=int, default=4, help="recortes por chip dorado y epoca")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="limita pasos por epoca (prueba rapida)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="run")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device if a.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Dispositivo:", dev)
    out = pathlib.Path(os.environ.get("RUNS_DIR", "runs")) / a.name
    out.mkdir(parents=True, exist_ok=True)

    # ---- datos
    only = set(open(a.subset).read().split()) if a.subset else None
    extra = "gold_S2.npy" if a.s2 else None
    if (a.s2 or only) and a.sets != "gold":
        raise SystemExit("--s2 y --subset solo se admiten con --sets gold")
    parts = []
    if a.sets in ("gold", "gold+weak"):
        parts.append(FloodDataset(a.data, "gold", indices(a.data, "gold", ["train"], only_chips=only),
                                  a.crop, True, a.gold_repeat, extra=extra))
    if a.sets in ("weak", "gold+weak"):  # los chips debiles de Bolivia nunca se usan
        parts.append(FloodDataset(a.data, "weak", indices(a.data, "weak", exclude_events=["Bolivia"]),
                                  a.crop, True, 1))
    train_ds = ConcatDataset(parts)
    mk = lambda split: DataLoader(FloodDataset(a.data, "gold", indices(a.data, "gold", [split], only_chips=only), extra=extra),
                                  batch_size=4, shuffle=False, num_workers=0)
    loaders = {s: mk(s) for s in ["valid", "test", "bolivia"]}
    train_dl = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                          drop_last=True, persistent_workers=a.workers > 0)
    print(f"Entrenamiento: {len(train_ds)} recortes/epoca | valid {len(loaders['valid'].dataset)} "
          f"| test {len(loaders['test'].dataset)} | bolivia {len(loaders['bolivia'].dataset)}")

    # ---- modelo
    model = UNet(8 if a.s2 else 2, a.base, a.depth, skip=not a.no_skip).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
        print("Pesos iniciales cargados de", a.init)
    print(f"U-Net: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parametros")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = get_loss(a.loss)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, log = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = n = 0
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); n += 1
            if a.max_steps and step + 1 >= a.max_steps:
                break
        sched.step()
        v = evaluate(model, loaders["valid"], dev)
        log.append(dict(epoch=ep, train_loss=tot / n, **{f"valid_{k}": v[k] for k in ["IoU", "F1", "precision", "recall"]}))
        print(f"ep {ep:3d} | loss {tot / n:.4f} | valid IoU {v['IoU']:.4f} F1 {v['F1']:.4f} "
              f"P {v['precision']:.3f} R {v['recall']:.3f} | {time.time() - t0:.0f}s")
        if v["IoU"] > best:
            best = v["IoU"]; torch.save(model.state_dict(), out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)

    # ---- evaluacion final con la mejor epoca (seleccionada en valid)
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    res = {s: evaluate(model, dl, dev) for s, dl in loaders.items()}
    res["args"] = vars(a)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print("\n=== Resultado final (mejor epoca por valid) ===")
    for s in ["valid", "test", "bolivia"]:
        r = res[s]
        print(f"{s:8s} IoU {r['IoU']:.4f} | mIoU/chip {r['mIoU_chip']:.4f} | F1 {r['F1']:.4f} | "
              f"P {r['precision']:.3f} R {r['recall']:.3f} | omision {r['omission']:.3f} comision {r['commission']:.3f}")


if __name__ == "__main__":
    main()
