"""Dataset de PyTorch sobre los .npy generados por 02_preprocess_sen1floods11.py"""
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def indices(root, name, splits=None, exclude_events=(), only_chips=None):
    """Posiciones (columna i) de los chips de `name` ('gold'|'weak') que cumplen el filtro."""
    df = pd.read_csv(pathlib.Path(root) / f"{name}_index.csv")
    m = np.ones(len(df), bool)
    if splits is not None:
        m &= df.split.isin(splits).values
    if len(exclude_events):
        m &= ~df.event.isin(exclude_events).values
    if only_chips is not None:  # restringe a una lista de chips (p. ej. los libres de nubes)
        m &= df.chip.isin(only_chips).values
    return df.i.values[m]


class FloodDataset(Dataset):
    """
    Devuelve x (2,h,w) float32 y y (h,w) int64 con valores {-1 ignorar, 0 no agua, 1 agua}.
    crop    : lado del recorte aleatorio (None = chip completo de 512)
    augment : volteos y rotaciones de 90 grados
    repeat  : repite el conjunto (mas recortes por chip en cada epoca)
    extra   : archivo .npy con canales adicionales (p. ej. 'gold_S2.npy', Sentinel-2) que se concatenan a VV,VH
    """

    def __init__(self, root, name, idx, crop=None, augment=False, repeat=1, extra=None):
        self.root, self.name = pathlib.Path(root), name
        self.idx, self.crop, self.augment, self.repeat = np.asarray(idx), crop, augment, repeat
        self.extra = extra
        self._E = None
        self._X = self._Y = None  # se abren dentro de cada worker (evita copiar el memmap)

    def __len__(self):
        return len(self.idx) * self.repeat

    def _open(self):
        if self._X is None:
            self._X = np.load(self.root / f"{self.name}_X.npy", mmap_mode="r")
            self._Y = np.load(self.root / f"{self.name}_Y.npy", mmap_mode="r")
            if self.extra:
                self._E = np.load(self.root / self.extra, mmap_mode="r")

    def __getitem__(self, k):
        self._open()
        i = self.idx[k % len(self.idx)]
        x = np.array(self._X[i], dtype=np.float32)
        y = np.array(self._Y[i], dtype=np.int64)
        if self._E is not None:
            x = np.concatenate([x, np.array(self._E[i], dtype=np.float32)], axis=0)
        if self.crop:
            h = x.shape[1]
            r, c = (int(v) for v in torch.randint(0, h - self.crop + 1, (2,)))
            x, y = x[:, r:r + self.crop, c:c + self.crop], y[r:r + self.crop, c:c + self.crop]
        if self.augment:
            if torch.rand(1) < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if torch.rand(1) < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            n = int(torch.randint(0, 4, (1,)))
            x, y = np.rot90(x, n, axes=(1, 2)), np.rot90(y, n, axes=(0, 1))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
