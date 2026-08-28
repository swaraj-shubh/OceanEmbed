"""M2 -- plain U-Net: 7 surface channels [B,7,96,176] -> 15 depth levels [B,15,96,176].

No attention, no temporal context yet; those are M3/M4 and only get added once this
beats M0 (CLAUDE.md sec.3). Kept deliberately small -- the region is 96x176.
"""
import torch
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=7, out_ch=15, base=32, depth=3):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([block(c_in, c) for c_in, c in zip([in_ch] + chs[:-1], chs)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        for u, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([u(x), s], 1))
        return self.head(x)


def masked_mse(pred, true, mask):
    """Land and missing target cells never contribute (CLAUDE.md rule 5)."""
    n = mask.sum()
    assert n > 0, "batch has no valid target cells"
    return ((pred - true) ** 2 * mask).sum() / n


if __name__ == "__main__":
    torch.manual_seed(0)
    net = UNet()
    x = torch.randn(2, 7, 96, 176)
    y = net(x)
    assert y.shape == (2, 15, 96, 176), y.shape
    print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    m = torch.ones_like(y, dtype=torch.bool); m[:, :, :3, :3] = False
    assert torch.isclose(masked_mse(y, y + 1, m), torch.tensor(1.0))
    junk = y.detach().clone(); junk[:, :, :3, :3] = 1e6      # garbage only where masked
    assert masked_mse(junk, y.detach(), m) < 1e-6

    # must be able to overfit a single sample (CLAUDE.md sec.13).
    # Target is a smooth field, as real temperature fields are -- not white noise.
    xb = torch.randn(1, 7, 96, 176)
    yb = torch.nn.functional.interpolate(torch.randn(1, 15, 6, 11), size=(96, 176),
                                         mode="bilinear", align_corners=False)
    ones = torch.ones_like(yb, dtype=torch.bool)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = masked_mse(net(xb), yb, ones)
        loss.backward(); opt.step()
        first = float(loss.detach()) if first is None else first
    last = float(loss.detach())
    assert last < 0.05 * first, f"cannot overfit one sample: {first:.3f} -> {last:.3f}"
    print(f"unet self-check OK -- overfit loss {first:.3f} -> {last:.4f}")
