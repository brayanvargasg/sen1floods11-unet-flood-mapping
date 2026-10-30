"""Perdidas y metricas con enmascarado de pixeles sin dato (etiqueta -1)."""
import torch
import torch.nn.functional as F


def bce_loss(logits, y):
    m = (y >= 0).float()
    l = F.binary_cross_entropy_with_logits(logits[:, 0], (y == 1).float(), reduction="none")
    return (l * m).sum() / m.sum().clamp(min=1)


def dice_loss(logits, y, eps=1.0):
    m = (y >= 0).float()
    p = torch.sigmoid(logits[:, 0]) * m
    t = (y == 1).float()
    return 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def get_loss(name):
    return {"bce": bce_loss,
            "dice": dice_loss,
            "bce_dice": lambda lg, y: bce_loss(lg, y) + dice_loss(lg, y)}[name]


@torch.no_grad()
def evaluate(model, loader, device):
    """Metricas de la clase 'agua' sobre pixeles validos (umbral 0.5 <=> logit > 0)."""
    model.eval()
    tp = fp = fn = tn = 0
    chip_iou = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)[:, 0] > 0
        valid, t = y >= 0, y == 1
        tp += int((pred & t & valid).sum()); fp += int((pred & ~t & valid).sum())
        fn += int((~pred & t & valid).sum()); tn += int((~pred & ~t & valid).sum())
        inter = (pred & t & valid).sum((1, 2)).float()
        union = ((pred | t) & valid).sum((1, 2)).float()
        chip_iou += (inter[union > 0] / union[union > 0]).tolist()
    d = lambda a, b: a / b if b else 0.0
    prec, rec = d(tp, tp + fp), d(tp, tp + fn)
    return dict(IoU=d(tp, tp + fp + fn), mIoU_chip=sum(chip_iou) / max(len(chip_iou), 1),
                precision=prec, recall=rec, F1=d(2 * prec * rec, prec + rec),
                omission=1 - rec, commission=1 - prec, tp=tp, fp=fp, fn=fn, tn=tn)
