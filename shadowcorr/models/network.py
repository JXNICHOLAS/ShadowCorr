"""
Sparse voxel CNN with multi-head local attention for instance embedding.

Architecture: sparse 3D convolutions (TorchSparse) + k-NN restricted
multi-head self-attention layers → per-voxel instance embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse import SparseTensor
from torchsparse.nn import BatchNorm, Conv3d, ReLU

import warnings

try:
    from torch_cluster import knn as torch_knn
    USE_TORCH_CLUSTER = True
except ImportError:
    from scipy.spatial import cKDTree
    USE_TORCH_CLUSTER = False
    warnings.warn("torch_cluster not installed; falling back to CPU cKDTree for k-NN (slower)")


class MultiHeadLocalAttention(nn.Module):
    """k-NN restricted multi-head attention on voxel features."""

    def __init__(self, in_dim, k=16, num_heads=4):
        super().__init__()
        assert in_dim % num_heads == 0, (
            f"in_dim ({in_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.k = k
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.to_q = nn.Linear(in_dim, in_dim, bias=False)
        self.to_k = nn.Linear(in_dim, in_dim, bias=False)
        self.to_v = nn.Linear(in_dim, in_dim, bias=False)
        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, feats, coords):
        N, C = feats.shape
        k_actual = min(self.k, N)
        H, D = self.num_heads, self.head_dim

        if USE_TORCH_CLUSTER:
            edge_index = torch_knn(coords, coords, k=k_actual)
            knn_idx = edge_index[1].view(N, k_actual)
        else:
            tree = cKDTree(coords.cpu().numpy())
            knn_idx = tree.query(coords.cpu().numpy(), k=k_actual)[1]
            knn_idx = torch.from_numpy(knn_idx).to(feats.device)

        q = self.to_q(feats).view(N, H, D)
        k_feat = self.to_k(feats).view(N, H, D)
        v = self.to_v(feats).view(N, H, D)

        knn_idx_expanded = knn_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, D)
        k_feat_expanded = k_feat.unsqueeze(0).expand(N, -1, -1, -1)
        v_expanded = v.unsqueeze(0).expand(N, -1, -1, -1)
        k_neighbors = torch.gather(k_feat_expanded, 1, knn_idx_expanded)
        v_neighbors = torch.gather(v_expanded, 1, knn_idx_expanded)

        q = q.unsqueeze(1)
        attn_scores = (q * k_neighbors).sum(-1) / (D ** 0.5)
        attn_scores = attn_scores.permute(0, 2, 1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.permute(0, 2, 1).unsqueeze(-1)
        out = (attn_weights * v_neighbors).sum(1).reshape(N, C)
        return self.out_proj(out)


class RockInstanceNetSparse(nn.Module):
    """
    Sparse CNN backbone: two sparse 3D conv blocks each followed by
    multi-head local attention, then a 1×1 projection to instance_embed_dim.
    """

    def __init__(
        self,
        in_channels=16,
        instance_embed_dim=32,
        attn_k=16,
        attn_k1=None,
        attn_k2=None,
        num_heads1=4,
        num_heads2=8,
        use_input_bn=None,
    ):
        super().__init__()
        self.instance_embed_dim = instance_embed_dim
        self.in_channels = in_channels
        k1 = attn_k1 if attn_k1 is not None else attn_k
        k2 = attn_k2 if attn_k2 is not None else attn_k

        if use_input_bn is None:
            self._use_input_bn = in_channels >= 16
        else:
            self._use_input_bn = use_input_bn

        if self._use_input_bn:
            self.input_bn = BatchNorm(in_channels)
            self.input_proj = None
        else:
            self.input_bn = None
            if in_channels < 16:
                self.input_proj = nn.Sequential(
                    nn.Linear(in_channels, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, in_channels),
                )
            else:
                self.input_proj = None

        self.conv1 = Conv3d(in_channels, 32, kernel_size=3, stride=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.act1 = ReLU(True)
        self.local_attn1 = MultiHeadLocalAttention(32, k=k1, num_heads=num_heads1)

        self.conv2 = Conv3d(32, 64, kernel_size=5, stride=1, bias=False)
        self.bn2 = BatchNorm(64)
        self.act2 = ReLU(True)
        self.local_attn2 = (
            MultiHeadLocalAttention(64, k=k2, num_heads=num_heads2) if k2 > 0 else None
        )

        self.conv3 = Conv3d(64, self.instance_embed_dim, kernel_size=1, stride=1, bias=True)

    def forward(self, x: SparseTensor):
        if self._use_input_bn:
            x = self.input_bn(x)
        elif self.input_proj is not None:
            feats = self.input_proj(x.feats)
            x = SparseTensor(coords=x.coords, feats=feats)

        x = self.act1(self.bn1(self.conv1(x)))
        feats = x.feats
        coords = x.coords[:, 1:4].float()
        x.feats = self.local_attn1(feats, coords)

        x = self.act2(self.bn2(self.conv2(x)))
        if self.local_attn2 is not None:
            feats = x.feats
            coords = x.coords[:, 1:4].float()
            x.feats = self.local_attn2(feats, coords)

        x = self.conv3(x)
        return x.feats
