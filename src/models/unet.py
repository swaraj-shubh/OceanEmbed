"""M2 U-Net and M3 OceanEmbed: 7 surface channels [B,7,96,176] -> 15 depths [B,15,96,176].

M2 is the plain U-Net. M3 adds self-attention over the bottleneck, and that attended
bottleneck IS the OceanEmbed latent -- the learned representation the project is named
for. Temporal context is M4 and is not here yet (CLAUDE.md sec.3).
"""
import torch
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SpatialAttention(nn.Module):
    """Multi-head self-attention over the bottleneck's spatial positions.

    At base=32, depth=3 the bottleneck is [B, 256, 12, 22] -- 264 tokens, so full
    attention is cheap here and would not be at full resolution. Every cell can attend to
    every other, which is the point: subsurface structure at one location depends on the
    surface state of the whole basin (eddies, coastal upwelling, the monsoon gyre), not
    just on the receptive field a stack of 3x3 convolutions happens to reach.

    The positional embedding is learned and tied to the fixed 96x176 region -- this is a
    regional model on a frozen grid, so absolute position is a real signal (the Arabian
    Sea and the Bay of Bengal behave differently), not a nuisance to be made invariant.
    """

    def __init__(self, ch, heads=4, hw=(12, 22)):
        super().__init__()
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.norm = nn.LayerNorm(ch)
        self.pos = nn.Parameter(torch.zeros(1, hw[0] * hw[1], ch))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)                  # [B, HW, C]
        assert t.shape[1] == self.pos.shape[1], (
            f"attention was built for {self.pos.shape[1]} positions, got {t.shape[1]} -- "
            "hw must match the bottleneck size for this grid and depth")
        t = self.norm(t + self.pos)
        t = t + self.attn(t, t, t, need_weights=False)[0]  # residual: never worse than M2
        return t.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, in_ch=7, out_ch=15, base=32, depth=3, attn=False, heads=4,
                 bottleneck_hw=(12, 22)):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([block(c_in, c) for c_in, c in zip([in_ch] + chs[:-1], chs)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.attn = SpatialAttention(chs[-1], heads, bottleneck_hw) if attn else None

    def embed(self, x):
        """Encoder + fusion. Returns (latent, skips) -- the latent is OceanEmbed."""
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = self.pool(x)
        return (self.attn(x) if self.attn is not None else x), skips

    def forward(self, x):
        x, skips = self.embed(x)
        for u, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([u(x), s], 1))
        return self.head(x)


class OceanEmbed(UNet):
    """M3 -- the U-Net with attention fusion switched on. Named separately so the config,
    the checkpoint and the ablation table all say which stage produced a number."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **{**kw, "attn": True})


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

    # M3: same contract, plus the latent the model is named for
    torch.manual_seed(0)
    oe = OceanEmbed()
    assert oe(x).shape == (2, 15, 96, 176)
    lat, skips = oe.embed(x)
    assert lat.shape == (2, 256, 12, 22), lat.shape
    assert len(skips) == 3
    print(f"oceanembed params {sum(p.numel() for p in oe.parameters()) / 1e6:.2f}M "
          f"(+{(sum(p.numel() for p in oe.parameters()) - sum(p.numel() for p in net.parameters())) / 1e3:.0f}k)")

    # attention must actually mix positions: perturbing one cell has to move the others,
    # otherwise it has silently collapsed to a per-pixel transform and M3 == M2.
    oe.eval()
    with torch.no_grad():
        a = oe.attn(torch.zeros(1, 256, 12, 22))
        z = torch.zeros(1, 256, 12, 22); z[0, :, 0, 0] = 5.0
        b = oe.attn(z)
        # Control, and it must run under the SAME no_grad: torch picks a different
        # attention kernel when grad is required, and the two differ in the last bits.
        same = (oe.attn(torch.zeros(1, 256, 12, 22)) - a).abs().max()
    moved = (a - b).abs().flatten(2).mean(1).reshape(12, 22)
    # A far-away cell must react to the perturbed one. The threshold is loose on purpose:
    # untrained, softmax over 264 positions spreads a single-token change to ~1/264 of its
    # size and out_proj starts small, so this is ~4e-8 at init and grows with training.
    # The control below is what makes it a real test -- identical inputs must give exactly
    # zero, so a nonzero value here can only have come from cross-position mixing.
    assert float(same) == 0.0, "attention is not deterministic; the control is meaningless"
    assert moved[3, 7] > 0, "attention did not propagate information between positions"

    torch.manual_seed(0)
    oe2 = OceanEmbed()
    opt = torch.optim.Adam(oe2.parameters(), 1e-3)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = masked_mse(oe2(xb), yb, ones)
        loss.backward(); opt.step()
        first = float(loss.detach()) if first is None else first
    last = float(loss.detach())
    assert last < 0.05 * first, f"oceanembed cannot overfit: {first:.3f} -> {last:.3f}"
    print(f"oceanembed self-check OK -- overfit loss {first:.3f} -> {last:.4f}")
