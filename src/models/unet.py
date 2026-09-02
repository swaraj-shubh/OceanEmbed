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


class ConvLSTMCell(nn.Module):
    """One convolutional LSTM step. Gates come from a single conv over [x, h], which is
    the standard formulation and a quarter of the convs of four separate gate convs."""

    def __init__(self, ch, k=3):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(2 * ch, 4 * ch, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class OceanEmbedTemporal(UNet):
    """M4 -- encoder per day, ConvLSTM over the 7-day sequence, then the U-Net decoder.

    The recurrence runs at the BOTTLENECK, not at full resolution: 12x22 costs a fraction
    of 96x176 per step, and the temporal signal being asked about here -- how the last week
    of surface forcing set up today's subsurface structure -- is a basin-scale thing, not a
    per-pixel one. Skips come from the LAST frame, so fine spatial detail is today's while
    the bottleneck carries the week.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cell = ConvLSTMCell(self.head.in_channels * 2 ** (len(self.down) - 1))

    def embed(self, x):
        assert x.dim() == 5, f"expected [B, T, C, H, W], got {tuple(x.shape)}"
        b, t = x.shape[:2]
        h = c = None
        for i in range(t):
            z, skips = super().embed(x[:, i])          # attention, if any, applies per frame
            if h is None:
                h = torch.zeros_like(z); c = torch.zeros_like(z)
            h, c = self.cell(z, (h, c))
        return h, skips                                # skips are the final frame's


class OceanEmbed(UNet):
    """M3 -- the U-Net with attention fusion switched on. Named separately so the config,
    the checkpoint and the ablation table all say which stage produced a number."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **{**kw, "attn": True})


def masked_mse(pred, true, mask, grad_weight=0.0, depth_weight=None):
    """Land and missing target cells never contribute (CLAUDE.md rule 5).

    `depth_weight` is a [15] tensor re-weighting each level's contribution. Note that plain
    MSE in degC ALREADY matches the reported blended metric almost exactly -- the blended
    score is an n-weighted RMS across depths with roughly equal n per level, i.e. the mean
    per-depth MSE. So depth weighting is a deliberate trade, not a free win: inverse-variance
    weights (1/sd^2 from the frozen train stats: 0.38 at 100 m rising to 2.53 at 1000 m)
    push effort out of the thermocline and into the deep, which is where climatology still
    beats us. Expect the blended number to worsen and 500-1000 m to improve.

    `grad_weight` adds a penalty on the error in the VERTICAL profile shape -- the
    level-to-level differences rather than the levels themselves. Plain MSE is happy with a
    profile that is smooth and slightly wrong in the thermocline, which is exactly the
    failure we measured: a stable +0.85 degC bias at 100 m, i.e. the model smearing the
    temperature drop instead of placing it sharply. Differences are taken per level, not
    per metre: dividing by dz would make the 5 m spacings near the surface dominate a term
    meant to be about the thermocline.
    """
    n = mask.sum()
    assert n > 0, "batch has no valid target cells"
    if depth_weight is None:
        loss = ((pred - true) ** 2 * mask).sum() / n
    else:
        wm = mask * depth_weight.view(1, -1, 1, 1).to(pred.device)
        loss = ((pred - true) ** 2 * wm).sum() / wm.sum()
    if grad_weight:
        gm = mask[:, 1:] & mask[:, :-1]          # both levels of the pair must be valid
        if gm.any():
            gp = pred[:, 1:] - pred[:, :-1]
            gt = true[:, 1:] - true[:, :-1]
            loss = loss + grad_weight * ((gp - gt) ** 2 * gm).sum() / gm.sum()
    return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    net = UNet()
    x = torch.randn(2, 7, 96, 176)
    y = net(x)
    assert y.shape == (2, 15, 96, 176), y.shape
    print(f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    m = torch.ones_like(y, dtype=torch.bool); m[:, :, :3, :3] = False
    assert torch.isclose(masked_mse(y, y + 1, m), torch.tensor(1.0))
    # a constant offset is pure bias: it has zero gradient error, so the extra term is 0
    assert torch.isclose(masked_mse(y, y + 1, m, grad_weight=1.0), torch.tensor(1.0)),         "gradient term must ignore a constant offset"
    # a profile with the right values but the wrong shape must cost more
    warped = y.detach().clone(); warped[:, 7] += 2.0; warped[:, 8] -= 2.0
    assert masked_mse(warped, y.detach(), m, grad_weight=1.0) >            masked_mse(warped, y.detach(), m, grad_weight=0.0), "gradient term is inert"
    junk = y.detach().clone(); junk[:, :, :3, :3] = 1e6      # garbage only where masked
    assert masked_mse(junk, y.detach(), m) < 1e-6

    # depth weighting: uniform weights must be a no-op, and a zero weight must silence its
    # level completely. The unweighted control is what makes the second one a real test.
    assert torch.isclose(masked_mse(y, y + 1, m, depth_weight=torch.ones(15)),
                         masked_mse(y, y + 1, m)), "uniform depth weights must be a no-op"
    w_deep = torch.zeros(15); w_deep[-1] = 1.0
    err = y.detach().clone(); err[:, :-1] += 5.0            # error only at levels 0..13
    assert masked_mse(err, y.detach(), m, depth_weight=w_deep) < 1e-6, \
        "zero-weighted levels still contributed"
    assert masked_mse(err, y.detach(), m) > 1.0, "control: unweighted must see the error"

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

    # M4: the whole claim is that history matters, so the test is that history CHANGES the
    # answer. If shuffling the earlier days left the output alone, the ConvLSTM would be an
    # expensive way to look at the last frame, and every M4-vs-M2 number would be noise.
    torch.manual_seed(0)
    tm = OceanEmbedTemporal().eval()
    xt = torch.randn(2, 7, 7, 96, 176)
    # Test at the BOTTLENECK, where the recurrence actually lives. Untrained, the LSTM
    # starts near-neutral -- forget gates sit around 0.5, so six days back is already
    # damped by 0.5^6 -- and the decoder's skips come from the last frame, so the effect is
    # ~1e-7 at the output and invisible there. That is initialisation, not inertness: the
    # bottleneck moves, and training is what grows the dependence.
    with torch.no_grad():
        lat_a, _ = tm.embed(xt)
        no_hist = xt.clone(); no_hist[:, :6] = 0.0            # same last day, no history
        lat_b, _ = tm.embed(no_hist)
        lat_same, _ = tm.embed(xt)
    assert float((lat_a - lat_same).abs().max()) == 0.0, "not deterministic; control is void"
    assert float((lat_a - lat_b).abs().max()) > 1e-6,         "erasing six days of history left the bottleneck untouched -- the LSTM is inert"
    assert tm(xt).shape == (2, 15, 96, 176)
    print(f"temporal self-check OK -- params {sum(p.numel() for p in tm.parameters())/1e6:.2f}M, "
          f"history moves the bottleneck by {float((lat_a - lat_b).abs().max()):.2e}")
