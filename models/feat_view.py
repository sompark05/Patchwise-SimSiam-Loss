# models/modules/feat_view.py
import torch
import torch.nn.functional as F

def make_feat_view(f, max_shift=4, drop_p=0.25, noise_std=0.03, crop_ratio=0.8):
    """
    Make a second view in feature space with tiny cost, but let gradients flow.
    """
    B, C, H, W = f.shape
    device = f.device

    # (a) small integer spatial roll
    dx = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device).item())
    dy = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device).item())
    f2 = torch.roll(f, shifts=(dy, dx), dims=(2, 3))

    # (b) per-sample channel dropout
    if drop_p > 0:
        mask = (torch.rand(B, C, 1, 1, device=device) > drop_p).float()
        f2 = f2 * mask

    # (c) tiny gaussian noise
    if noise_std > 0:
        f2 = f2 + noise_std * torch.randn_like(f2)

    # (d) light crop-resize
    if 0 < crop_ratio < 1.0 and H >= 12 and W >= 12:
        h = max(8, int(H * crop_ratio))
        w = max(8, int(W * crop_ratio))
        y0 = int(torch.randint(0, H - h + 1, (1,), device=device).item())
        x0 = int(torch.randint(0, W - w + 1, (1,), device=device).item())
        f2 = F.interpolate(f2[:, :, y0:y0 + h, x0:x0 + w],
                           size=(H, W), mode="bilinear", align_corners=False)
    return f2
