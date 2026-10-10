# ---- solo model: shared by training (train_solo.py) and inference (solo_driver.py) ----
# One source for both sides, so the model that is scored is the model that was trained.
# The build scripts paste this file into the training job and into the submission script.
import torch
import torch.nn as nn
import torch.nn.functional as F

SOLO_ARCHS = {"cnx": "convnext_tiny", "r50": "resnet50", "effb0": "efficientnet_b0"}


def solo_timm_name(arch):
    """`cnx` -> `convnext_tiny`; `timm:<name>` passes any timm model through."""
    if arch.startswith("timm:"):
        return arch[5:]
    if arch not in SOLO_ARCHS:
        raise ValueError(f"unknown arch {arch!r}; use one of {sorted(SOLO_ARCHS)} or timm:<name>")
    return SOLO_ARCHS[arch]


class SoloHead(nn.Module):
    """One attention query per finding over every (series slot, slice window) token.

    The E3 head attended over the six slots only, and each slot was a single three-slice
    image. Here a slot contributes one token per window, so a finding can pick the part of
    the stack where it is visible (a meniscal tear sits on a few slices, not on all).
    Parameter names match the E3 `SlotHead`, so E3 weights load with `win_emb` left at zero.
    """
    def __init__(self, dim, n_slot, n_win, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.win_emb = nn.Parameter(torch.zeros(n_win, hidden))
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask):
        # x [B, S, G, D]; mask [B, S] (1 = the slot holds a series)
        B, S, G, _ = x.shape
        h = self.proj(x) + self.slot_emb[None, :, None, :] + self.win_emb[None, None, :G, :]
        h = h.reshape(B, S * G, -1)
        m = mask[:, :, None].expand(B, S, G).reshape(B, S * G)
        att = torch.einsum("bth,oh->bot", h, self.query) / self.hidden ** 0.5
        att = att.masked_fill(m.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bot,bth->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class SoloModel(nn.Module):
    """A CNN encoder per three-slice image + the per-finding attention head."""
    # The encoder batch is the number of present slots, which differs every step. Each new
    # batch size costs the GPU library a one-off set-up (bench v2: ~0.6 s per batch, more
    # than the batch itself). Rounding the batch up to a multiple of BUCKET leaves a handful
    # of sizes. The filler rows are copies of real images and their outputs are dropped.
    BUCKET = 24
    def __init__(self, timm_name, n_slot, n_win, n_out, pretrained=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        self.head = SoloHead(self.backbone.num_features, n_slot, n_win, n_out)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        # imgs uint8 [B, S, G, 3, H, W]; mask [B, S]
        B, S, G = imgs.shape[:3]
        present = mask > 0.5
        dim = self.backbone.num_features
        if not bool(present.any()):
            feat = torch.zeros(B, S, G, dim, device=imgs.device, dtype=self.head.out.weight.dtype)
            return self.head(feat, mask)
        # Absent slots are masked out of the attention, so encoding them is wasted work:
        # a quarter of all slots are empty in this corpus.
        x = imgs[present].reshape(-1, *imgs.shape[3:])
        n = x.shape[0]
        fill = (-n) % self.BUCKET if self.BUCKET else 0
        if fill:
            x = torch.cat([x, x[torch.arange(fill, device=x.device) % n]])
        x = x.float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        f = self.backbone((x - self.mean) / self.std)[:n]
        feat = f.new_zeros(B, S, G, dim)
        feat[present] = f.reshape(-1, G, dim)
        return self.head(feat, mask)


def solo_windows(cache_u8):
    """Cache slices [B, S, G*3, H, W] -> windows [B, S, G, 3, H, W] (consecutive triples)."""
    B, S, C, H, W = cache_u8.shape
    return cache_u8.reshape(B, S, C // 3, 3, H, W)


def solo_logits(model, cache_u8, mask, img_size, mode):
    """Logits [B, n_out] for a batch of cached studies.

    mode `all`: every window is a token of one forward pass (how `all` members train).
    mode `one`: each window is scored on its own and the logits are averaged (E3 members,
    which were trained on one window at a time).
    """
    x = solo_windows(cache_u8)
    if mode == "all":
        return model(x, mask, img_size)
    if mode != "one":
        raise ValueError(f"unknown window mode {mode!r}")
    acc = 0
    for g in range(x.shape[2]):
        acc = acc + model(x[:, :, g:g + 1], mask, img_size)
    return acc / x.shape[2]
