# models/tiny_simsiam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TinySimSiamHead(nn.Module):
    """
    Minimal SimSiam-style head for feature-space alignment.
    One-sided: aligns p(fake) toward sg(z(real)).

    - Projection MLP: 3 layers
    - Prediction MLP: 2 layers
    - ASP-style weighting supported via asp_loss_mode and total_epochs/current_epoch
    """

    def __init__(self, opt, c: int, proj_dim: int = 512, pred_dim: int = 256):
        super().__init__()
        self.opt = opt
        self.total_epochs = opt.n_epochs + opt.n_epochs_decay

        self.c = c
        # 1×1 conv before projection
        self.proj_conv = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.proj_norm = nn.GroupNorm(num_groups=8, num_channels=c)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ----- projection MLP (3 layers) -----
        self.proj_mlp = nn.Sequential(
            nn.Linear(c, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=True),
        )

        # ----- prediction MLP (2 layers) -----
        self.pred_mlp = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.LayerNorm(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim, bias=True),
        )

        # zero-init predictor last layer
        nn.init.zeros_(self.pred_mlp[-1].weight)
        nn.init.zeros_(self.pred_mlp[-1].bias)

    
    def _project(self, f: torch.Tensor) -> torch.Tensor:
        h = self.proj_conv(f)              # [B,C,H,W]
        h = self.proj_norm(h)
        B, C, H, W = h.shape
        v = h.view(B, C, H*W).permute(0, 2, 1)   # [B, N, C]
        z = self.proj_mlp(v)                     # [B, N, D]  (LN is along last dim)
        return z

    def _asp_weights(self, current_epoch: int, x: torch.Tensor):
        scheduler, lookup = self.opt.asp_loss_mode.split('_')[:2]
        # Compute scheduling
        t = (current_epoch - 1) / self.total_epochs
        if scheduler == 'sigmoid':
            p = 1 / (1 + np.exp((t - 0.5) * 10))
        elif scheduler == 'linear':
            p = 1 - t
        elif scheduler == 'lambda':
            k = 1 - self.opt.n_epochs_decay / self.total_epochs
            m = 1 / (1 - k)
            p = m - m * t if t >= k else 1.0
        elif scheduler == 'zero':
            p = 1.0
        else:
            raise ValueError(f"Unrecognized scheduler: {scheduler}")
        # Weight lookups
        w0 = 1.0
        if lookup == 'top':
            x = torch.where(x > 0.0, x, torch.zeros_like(x))
            w1 = torch.sqrt(1 - (x - 1) ** 2)
        elif lookup == 'linear':
            w1 = torch.relu(x)
        elif lookup == 'bell':
            sigma, mu, sc = 1, 0, 4
            w1 = 1 / (sigma * np.sqrt(2 * torch.pi)) * torch.exp(-((x - 0.5) * sc - mu) ** 2 / (2 * sigma ** 2))
        elif lookup == 'uniform':
            w1 = torch.ones_like(x)
        else:
            raise ValueError(f"Unrecognized lookup: {lookup}")
        # Apply weights with schedule
        w = p * w0 + (1 - p) * w1
        # Normalize
        w = w / w.sum() * len(w)
        return w

    def forward(self, current_epoch: int, f_fake: torch.Tensor, f_real: torch.Tensor, paired=False) -> torch.Tensor:
        """
        Args:
            current_epoch: int
            f_fake, f_real: [B, C, H, W]
            paired: ignored (kept only for backward compatibility)
        """
        # project both branches (no no_grad here; stop-grad is applied later)
        zf = self._project(f_fake)   # [B, N, D]
        zr = self._project(f_real)   # [B, N, D]

        B, N, D = zf.shape

        K = min(getattr(self.opt, "num_patches", 256), N)

        # sample same K indices across batch
        idx = torch.randint(0, N, (K,), device=zf.device)
        zf_k = zf[:, idx, :]         # [B, K, D]
        zr_k = zr[:, idx, :]         # [B, K, D]

        # predictors for both branches
        pf = self.pred_mlp(zf_k.reshape(B * K, D)).view(B, K, D)  # fake branch
        pr = self.pred_mlp(zr_k.reshape(B * K, D)).view(B, K, D)  # real branch

        # normalize
        pf = F.normalize(pf, dim=-1, eps=1e-12)
        pr = F.normalize(pr, dim=-1, eps=1e-12)
        zf_k_n = F.normalize(zf_k, dim=-1, eps=1e-12)
        zr_k_n = F.normalize(zr_k, dim=-1, eps=1e-12)

        # symmetric SimSiam:
        # L = -0.5 * [cos(p_fake, sg(z_real)) + cos(p_real, sg(z_fake))]
        cos_fr = (pf * zr_k_n.detach()).sum(dim=-1)   # [B, K] fake -> real
        cos_rf = (pr * zf_k_n.detach()).sum(dim=-1)   # [B, K] real -> fake
        cos = 0.5 * (cos_fr + cos_rf)                 # [B, K]
        per = -cos                                    # per-patch loss

        # ASP per-patch (vectorized over B*K), now always used if mode != 'none'
        if self.opt.asp_loss_mode != "none":
            w = self._asp_weights(current_epoch, cos.reshape(-1))  # [B*K]
            loss = (per.reshape(-1) * w).mean()
        else:
            loss = per.mean()

        return loss

