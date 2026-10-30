"""U-Net implementada desde cero (sin bibliotecas de segmentacion)."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    in_ch : canales de entrada (2 = VV, VH)
    base  : canales de la primera capa (se duplican en cada nivel)
    depth : numero de reducciones 2x (entrada debe ser divisible por 2**depth)
    skip  : False desactiva las conexiones de salto (para la ablacion de arquitectura)
    Devuelve logits (B,1,H,W); la sigmoide se aplica en la perdida / inferencia.
    """

    def __init__(self, in_ch=2, base=32, depth=4, skip=True):
        super().__init__()
        self.skip = skip
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.inc = ConvBlock(in_ch, ch[0])
        self.downs = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), ConvBlock(ch[i], ch[i + 1])) for i in range(depth)])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.decs = nn.ModuleList(
            [ConvBlock(ch[i] * (2 if skip else 1), ch[i]) for i in reversed(range(depth))])
        self.outc = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        x = feats.pop()  # cuello de botella
        for up, dec in zip(self.ups, self.decs):
            x = up(x)
            s = feats.pop()
            if self.skip:
                x = torch.cat([s, x], dim=1)
            x = dec(x)
        return self.outc(x)


if __name__ == "__main__":
    m = UNet()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parametros")
    print(m(torch.randn(2, 2, 256, 256)).shape)
