# stability.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

# -----------------------
# NodeNorm
# -----------------------
class NodeNorm(nn.Module):
    """
    Node-wise normalization: subtract mean (per node) and optionally scale by std.
    Useful to stabilize feature magnitude across layers.
    Input: x [N, D] or [B, N, D]
    If input is 2D: treat as [N, D]
    """
    def __init__(self, eps=1e-6, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        # x: [N, D] or [B, N, D]
        if x.dim() == 2:
            mean = x.mean(dim=1, keepdim=True)  # [N,1]
            std = x.std(dim=1, keepdim=True)    # [N,1]
            out = (x - mean) / (std + self.eps)
        elif x.dim() == 3:
            # [B,N,D] - compute per-node across feature dim
            mean = x.mean(dim=2, keepdim=True)  # [B,N,1]
            std = x.std(dim=2, keepdim=True)
            out = (x - mean) / (std + self.eps)
        else:
            raise ValueError("NodeNorm expects 2D or 3D tensor")
        if self.affine:
            out = out * self.gamma + self.beta
        return out


# -----------------------
# PairNorm (from PairNorm paper)
# -----------------------
class PairNorm(nn.Module):
    """
    PairNorm: designed to reduce oversmoothing for GNNs.
    mode: 'PN' (original), 'PN-SI' (scale individually), 'PN-SCS' (scale and center separately)
    Implementation adapted for node features [N, D]
    """
    def __init__(self, mode='PN', scale=1.0, eps=1e-6):
        super().__init__()
        assert mode in ['PN', 'PN-SI', 'PN-SCS']
        self.mode = mode
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        # x: [N, D]
        if self.mode == 'PN':
            x = x - x.mean(dim=0, keepdim=True)           # center
            norm = torch.sqrt((x**2).sum(dim=1).mean())   # global norm
            x = self.scale * x / (norm + self.eps)
            return x
        elif self.mode == 'PN-SI':
            x = x - x.mean(dim=0, keepdim=True)
            norm = torch.sqrt((x**2).sum(dim=1, keepdim=True))  # per-node norm
            x = self.scale * x / (norm + self.eps)
            return x
        else:  # PN-SCS
            x = x - x.mean(dim=0, keepdim=True)
            col_norm = torch.sqrt((x**2).sum(dim=0, keepdim=True))
            x = self.scale * x / (col_norm + self.eps)
            return x


# -----------------------
# Label Smoothing CrossEntropy
# -----------------------
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross-entropy. Input logits [N, C], targets [N]
    smooth: 0.0 -> standard CE
    """
    def __init__(self, smoothing: float = 0.0, reduction='mean'):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, C], target: [N]
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        if self.smoothing == 0.0:
            loss = F.nll_loss(log_probs, target, reduction=self.reduction)
            return loss
        # create smoothed labels
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = - (true_dist * log_probs).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

