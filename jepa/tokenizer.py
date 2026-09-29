"""
4D BraTS Tokenizer & 3D Masking
---------------------------------
Converts [B, 4, H, W, D] MRI volumes into sequences of patch tokens,
then samples spatial multi-block masks for JEPA training.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, List


# ─── 3D Patch Embedding ───────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    Splits a [B, C, H, W, D] volume into non-overlapping 3D cubes and
    projects each P³ × C block to an embedding vector.

    Two projection heads share the same output space:
      • proj_multi  — Conv3d(4, D, P, stride=P) for the full 4-modality stack
                       (used by the spatial JEPA objective).
      • proj_single — Conv3d(1, D, P, stride=P) for a *single* modality
                       (used by the cross-modal objective).
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: int = 16,            # P; cube side length in voxels
        in_channels: int = 4,            # T1c, T1n, T2f, T2w
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        H, W, D = img_size
        assert H % patch_size == 0 and W % patch_size == 0 and D % patch_size == 0, \
            "Image dimensions must be divisible by patch_size"

        self.grid_size = (H // patch_size, W // patch_size, D // patch_size)
        self.num_patches = math.prod(self.grid_size)

        # Full 4-modality stack (spatial JEPA objective)
        self.proj_multi = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        # Shared single-modality projection (cross-modal objective).
        self.proj_single = nn.Conv3d(
            1, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, single_modality: bool = False) -> torch.Tensor:
        proj = self.proj_single if single_modality else self.proj_multi
        x = proj(x)                          # [B, D, H/P, W/P, D/P]
        x = x.flatten(2).transpose(1, 2)     # [B, N, D]
        return self.norm(x)


# ─── 3D Sinusoidal Positional Embedding ──────────────────────────────────────

class SinCos3DPosEmbed(nn.Module):
    """
    Factorised sinusoidal positional encoding for 3D grids.
    No learned parameters — robust to varying patch counts.
    """

    def __init__(self, embed_dim: int, grid_size: Tuple[int, int, int]):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        pos = self._build(embed_dim, grid_size)  # [N, D]
        self.register_buffer("pos_embed", pos.unsqueeze(0))  # [1, N, D]

    @staticmethod
    def _build(embed_dim: int, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        gh, gw, gd = grid_size
        d_per_axis = embed_dim // 3           # split dims across 3 axes
        d3 = embed_dim - 2 * d_per_axis

        def sincos(n: int, d: int) -> torch.Tensor:
            pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                -torch.arange(0, d, 2, dtype=torch.float32) * math.log(10000.0) / d
            )
            pe = torch.zeros(n, d)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:d // 2])
            return pe

        ph = sincos(gh, d_per_axis)   # [gh, d_per_axis]
        pw = sincos(gw, d_per_axis)   # [gw, d_per_axis]
        pd = sincos(gd, d3)           # [gd, d3]

        h_emb = ph.unsqueeze(1).unsqueeze(2).expand(gh, gw, gd, d_per_axis)
        w_emb = pw.unsqueeze(0).unsqueeze(2).expand(gh, gw, gd, d_per_axis)
        d_emb = pd.unsqueeze(0).unsqueeze(1).expand(gh, gw, gd, d3)

        combined = torch.cat([h_emb, w_emb, d_emb], dim=-1)  # [gh, gw, gd, D]
        return combined.reshape(-1, embed_dim)                 # [N, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to token sequence [B, N, D]."""
        return x + self.pos_embed


# ─── 3D Multi-Block Masking ───────────────────────────────────────────────────

class MultiBlock3DMasker:
    """Samples contiguous 3D block masks for JEPA."""

    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        mask_ratio: float = 0.65,
        num_blocks: int = 6,
        min_block_scale: float = 0.15,
        max_block_scale: float = 0.35,
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.33),
    ):
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)
        self.mask_ratio = mask_ratio
        self.num_blocks = num_blocks
        self.min_block_scale = min_block_scale
        self.max_block_scale = max_block_scale
        self.aspect_ratio_range = aspect_ratio_range

    def _sample_one_block(self, device: torch.device) -> List[int]:
        gh, gw, gd = self.grid_size

        scale = torch.empty(1).uniform_(self.min_block_scale, self.max_block_scale).item()
        ar_h = torch.empty(1).uniform_(*self.aspect_ratio_range).item()
        ar_w = torch.empty(1).uniform_(*self.aspect_ratio_range).item()

        bh = max(1, round(scale * gh * ar_h))
        bw = max(1, round(scale * gw * ar_w))
        bd = max(1, round(scale * gd))

        bh, bw, bd = min(bh, gh), min(bw, gw), min(bd, gd)

        sh = torch.randint(0, max(1, gh - bh + 1), (1,)).item()
        sw = torch.randint(0, max(1, gw - bw + 1), (1,)).item()
        sd = torch.randint(0, max(1, gd - bd + 1), (1,)).item()

        indices = []
        for hi in range(sh, sh + bh):
            for wi in range(sw, sw + bw):
                for di in range(sd, sd + bd):
                    indices.append(hi * gw * gd + wi * gd + di)
        return indices

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        context_list, target_list = [], []

        for _ in range(batch_size):
            target_set = set()
            attempts = 0
            while (
                len(target_set) / self.num_patches < self.mask_ratio
                and attempts < self.num_blocks * 5
            ):
                block_idx = self._sample_one_block(device)
                target_set.update(block_idx)
                attempts += 1

            target_idx = torch.tensor(sorted(target_set), device=device)
            all_idx = torch.arange(self.num_patches, device=device)

            target_mask = torch.zeros(self.num_patches, dtype=torch.bool, device=device)
            target_mask[target_idx] = True
            context_idx = all_idx[~target_mask]

            context_list.append(context_idx)
            target_list.append(target_idx)

        return context_list, target_list


# ─── Helper: gather tokens by index ──────────────────────────────────────────

def gather_tokens(tokens: torch.Tensor, indices: List[torch.Tensor]) -> torch.Tensor:
    gathered = []
    for b, idx in enumerate(indices):
        gathered.append(tokens[b][idx])  # [len_b, D]
    return gathered