"""
4D-JEPA Model Components
--------------------------
  • VisionTransformer3D  — shared encoder backbone
  • EMATargetEncoder     — exponential-moving-average copy, no grad
  • SpatialPredictor3D   — narrow transformer that maps context repr
                           + positional mask tokens → predicted target repr
  • CrossModalPredictor  — predicts one modality's repr from another
  • UNETRDecoder3D       — multi-scale segmentation decoder
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from tokenizer import PatchEmbed3D, SinCos3DPosEmbed, gather_tokens


# ─── Core Transformer Block ───────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_bias = None
        if key_padding_mask is not None:
            attn_bias = torch.zeros(B, 1, 1, N, device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        try:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        except Exception:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                bidirectional: bool = True) -> torch.Tensor:
        # `bidirectional` is accepted-but-unused here so VisionTransformer3D
        # can call attention and Mamba blocks through one shared code path.
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ─── Mamba (S6 selective SSM) mixer — alternative to Attention ───────────────

class MambaBlock(nn.Module):

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.d_inner = dim * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or max(dim // 16, 1)

        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is parameterised in log-space and kept negative via -exp(),
        # standard Mamba practice for a stable, always-decaying state.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))     # [d_inner, d_state]
        self.D = nn.Parameter(torch.ones(self.d_inner))  # skip/residual gain

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def _scan(self, x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
              B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Sequential selective-scan recurrence.
          x, delta: [B, L, d_inner]
          A:        [d_inner, d_state]           (negative)
          B, C:     [B, L, d_state]
        Returns y: [B, L, d_inner]
        """
        Bsz, L, d_inner = x.shape
        d_state = A.shape[1]

        deltaA = torch.exp(delta.unsqueeze(-1) * A)                       # [B,L,d_inner,d_state]
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # [B,L,d_inner,d_state]

        h = x.new_zeros(Bsz, d_inner, d_state)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
        return torch.stack(ys, dim=1)   # [B, L, d_inner]

    def _mixer(self, x: torch.Tensor, reverse: bool) -> torch.Tensor:
        """One directional pass. x: [B, L, dim] -> [B, L, dim]."""
        if reverse:
            x = x.flip(dims=[1])

        xz = self.in_proj(x)                        # [B,L,2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)

        # Causal depthwise conv for local context, matching original Mamba.
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :x_in.shape[1]]
        x_conv = F.silu(x_conv.transpose(1, 2))       # [B,L,d_inner]

        x_dbl = self.x_proj(x_conv)                   # [B,L,dt_rank+2*d_state]
        delta, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))       # [B,L,d_inner]
        A = -torch.exp(self.A_log)                    # [d_inner,d_state]

        y = self._scan(x_conv, delta, A, Bm, Cm)
        y = y + x_conv * self.D                        # skip connection
        y = y * F.silu(z)                               # gate
        out = self.out_proj(y)

        if reverse:
            out = out.flip(dims=[1])
        return out

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                bidirectional: bool = True) -> torch.Tensor:
        if key_padding_mask is not None:
            assert not bidirectional, (
                "Bidirectional Mamba scanning is not padding-safe — a backward "
                "scan hits padding before real tokens and contaminates every "
                "output. Pass bidirectional=False whenever key_padding_mask is set."
            )
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        fwd = self._mixer(x, reverse=False)
        if not bidirectional:
            return fwd
        bwd = self._mixer(x, reverse=True)
        return 0.5 * (fwd + bwd)


class MambaTransformerBlock(nn.Module):
    """Same pre-norm residual shape as TransformerBlock, but with the
    Attention mixer swapped for a bidirectional MambaBlock."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MambaBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                bidirectional: bool = True) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x), key_padding_mask=key_padding_mask,
                            bidirectional=bidirectional)
        x = x + self.mlp(self.norm2(x))
        return x


# ─── 3D Vision Transformer ───────────────────────────────────────────────────

class VisionTransformer3D(nn.Module):

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        block_type: str = "attention",   # "attention" or "mamba"
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        assert block_type in ("attention", "mamba"), f"unknown block_type: {block_type}"
        self.block_type = block_type

        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = SinCos3DPosEmbed(embed_dim, self.patch_embed.grid_size)

        if block_type == "attention":
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                MambaTransformerBlock(
                    embed_dim, mlp_ratio, drop_rate,
                    d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand,
                )
                for _ in range(depth)
            ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _tokenize(self, x: torch.Tensor, single_modality: bool = False) -> torch.Tensor:
        """Embed patches + add positional encoding. Returns [B, N, D]."""
        return self.pos_embed(self.patch_embed(x, single_modality=single_modality))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass on all tokens of the 4-modality stack. Used for fine-tuning."""
        tokens = self._tokenize(x, single_modality=False)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward_with_hidden_states(
        self, x: torch.Tensor, layer_indices: List[int]
    ) -> Dict[int, torch.Tensor]:
        tokens = self._tokenize(x, single_modality=False)
        wanted = set(layer_indices)
        last_idx = max(wanted)
        out: Dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if i in wanted:
                out[i] = self.norm(tokens) if i == last_idx else tokens
        return out

    def forward_single_modality(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass on all tokens of a SINGLE modality, e.g. x: [B, 1, H, W, D].
        Used by the cross-modal JEPA objective.
        """
        tokens = self._tokenize(x, single_modality=True)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def forward_subset(
        self,
        x: torch.Tensor,
        indices: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Efficient JEPA context encoding: only process visible tokens.
        """
        all_tokens = self._tokenize(x, single_modality=False)   # [B, N, D]

        ctx_tokens = gather_tokens(all_tokens, indices)  # list of [Nc_b, D]
        max_len = max(t.shape[0] for t in ctx_tokens)
        B, D = len(ctx_tokens), self.embed_dim
        device = x.device

        padded = torch.zeros(B, max_len, D, device=device)
        key_padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
        for b, t in enumerate(ctx_tokens):
            padded[b, :t.shape[0]] = t
            key_padding_mask[b, :t.shape[0]] = False

        for block in self.blocks:
            padded = block(padded, key_padding_mask=key_padding_mask, bidirectional=False)
        padded = self.norm(padded)

        out = [padded[b, :ctx_tokens[b].shape[0]] for b in range(B)]
        return out


# ─── EMA Target Encoder ───────────────────────────────────────────────────────

class EMATargetEncoder(nn.Module):
    """
    Maintains an exponential-moving-average copy of the context encoder.
        θ_target ← τ·θ_target + (1-τ)·θ_online
    τ is annealed from τ_start → τ_end over training steps.
    """

    def __init__(
        self,
        encoder: VisionTransformer3D,
        tau_start: float = 0.996,
        tau_end: float = 1.0,
        total_steps: int = 100_000,
    ):
        super().__init__()
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.step = 0

        self.encoder = copy.deepcopy(encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: VisionTransformer3D):
        frac = min(self.step / self.total_steps, 1.0)
        tau = self.tau_start + (self.tau_end - self.tau_start) * frac
        for p_ema, p_online in zip(self.encoder.parameters(), online.parameters()):
            p_ema.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)
        self.step += 1

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ─── Spatial Predictor ───────────────────────────────────────────────────────

class SpatialPredictor3D(nn.Module):
    """
    A narrow transformer that, given context representations + learnable
    mask tokens with positional embeddings of the target locations,
    predicts what the target encoder would output at those locations.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        predictor_dim: int = 384,
        num_heads: int = 6,
        depth: int = 6,
        grid_size: Tuple[int, int, int] = (8, 8, 8),
    ):
        super().__init__()
        self.predictor_dim = predictor_dim

        self.proj_in = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.pos_embed = SinCos3DPosEmbed(predictor_dim, grid_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, encoder_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_repr: List[torch.Tensor],
        context_indices: List[torch.Tensor],
        target_indices: List[torch.Tensor],
        num_total_patches: int,
    ) -> List[torch.Tensor]:
        B = len(context_repr)
        all_pos = self.pos_embed.pos_embed.squeeze(0)  # [N, D_pred]

        preds = []
        for b in range(B):
            ctx = self.proj_in(context_repr[b])       # [Nc, D_pred]
            ci = context_indices[b]                    # [Nc]
            ti = target_indices[b]                     # [Nt]

            Nc, Nt = ci.shape[0], ti.shape[0]

            ctx_seq = ctx + all_pos[ci]               # [Nc, D_pred]

            mask_tok = self.mask_token.expand(1, Nt, -1).squeeze(0)  # [Nt, D_pred]
            tgt_seq = mask_tok + all_pos[ti]            # [Nt, D_pred]

            seq = torch.cat([ctx_seq, tgt_seq], dim=0).unsqueeze(0)  # [1, Nc+Nt, D_pred]

            for block in self.blocks:
                seq = block(seq)
            seq = self.norm(seq)

            tgt_out = seq[0, Nc:]                      # [Nt, D_pred]
            preds.append(self.proj_out(tgt_out))       # [Nt, D_enc]

        return preds


# ─── Cross-Modal Predictor ───────────────────────────────────────────────────

class CrossModalPredictor(nn.Module):
    """
    Given the full-volume representation of *one* modality (e.g. T1c),
    predict the full-volume representation of *another* modality (e.g. T2f).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        predictor_dim: int = 384,
        num_heads: int = 6,
        depth: int = 4,
    ):
        super().__init__()
        self.proj_src = nn.Linear(embed_dim, predictor_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        self.cross_attn_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=predictor_dim, nhead=num_heads,
                dim_feedforward=predictor_dim * 4,
                batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, embed_dim)

        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(
        self,
        src_repr: torch.Tensor,   # [B, N, D_enc]
        tgt_repr: torch.Tensor,   # [B, N, D_enc]
    ) -> torch.Tensor:
        B, N, _ = src_repr.shape

        memory = self.proj_src(src_repr)          # [B, N, D_pred]
        queries = self.proj_src(tgt_repr)         # [B, N, D_pred]

        for block in self.cross_attn_blocks:
            queries = block(queries, memory)

        queries = self.norm(queries)
        return self.proj_out(queries)             # [B, N, D_enc]


# ─── UNETR-style Segmentation Decoder (NEW) ──────────────────────────────────

class ConvBlock3D(nn.Module):
    """Two 3x3x3 convs, each InstanceNorm + LeakyReLU. No resolution change."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock3D(nn.Module):
    """ConvTranspose3d upsample by 2x, followed by a ConvBlock3D."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class SkipProjector3D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, n_ups: int):
        super().__init__()
        if n_ups == 0:
            self.net = ConvBlock3D(in_ch, out_ch)
        else:
            chans = [in_ch] + [out_ch] * n_ups
            self.net = nn.Sequential(*[
                UpBlock3D(chans[i], chans[i + 1]) for i in range(n_ups)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNETRDecoder3D(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int, int],
        patch_size: int,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        self.grid_size = grid_size
        n = int(math.log2(patch_size))
        assert 2 ** n == patch_size, "patch_size must be a power of 2"
        self.n_stages = n

        bottleneck_ch = base_channels * (2 ** n)
        self.bottleneck_proj = ConvBlock3D(embed_dim, bottleneck_ch)

        # Output channels of each of the n upsample+fuse stages, halving
        # each time until reaching base_channels at full resolution.
        self.stage_channels = [base_channels * (2 ** (n - 1 - s)) for s in range(n)]

        # Skip projectors for the (n - 1) intermediate ViT layers. Stage
        # s's skip needs (s + 1) upsamples to reach that stage's resolution.
        self.skip_projectors = nn.ModuleList([
            SkipProjector3D(embed_dim, self.stage_channels[s], n_ups=s + 1)
            for s in range(n - 1)
        ])

        # Final stage's skip is the raw input volume, already full-res.
        self.input_stem = ConvBlock3D(in_channels, base_channels)

        self.ups = nn.ModuleList()
        self.fuses = nn.ModuleList()
        in_ch = bottleneck_ch
        for s in range(n):
            out_ch = self.stage_channels[s]
            self.ups.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2))
            self.fuses.append(ConvBlock3D(out_ch * 2, out_ch))
            in_ch = out_ch

        self.head = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def _grid(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        gh, gw, gd = self.grid_size
        return tokens.permute(0, 2, 1).reshape(B, D, gh, gw, gd)

    def forward(
        self,
        hidden_states: Dict[int, torch.Tensor],
        x_input: torch.Tensor,
        layer_indices: List[int],
    ) -> torch.Tensor:
        """
        hidden_states: {layer_idx: [B, N, D]}, from forward_with_hidden_states
        x_input:       [B, C_in, H, W, D] raw volume (full-res skip source)
        layer_indices: ascending list of length n_stages (n_stages - 1
                       intermediate skip layers, then the bottleneck last)
        """
        assert len(layer_indices) == self.n_stages, (
            f"expected {self.n_stages} layer indices, got {len(layer_indices)}"
        )

        feat = self.bottleneck_proj(self._grid(hidden_states[layer_indices[-1]]))

        skips = [
            self.skip_projectors[s](self._grid(hidden_states[layer_indices[s]]))
            for s in range(self.n_stages - 1)
        ]
        skips.append(self.input_stem(x_input))  # full-res skip, fused last

        for up, fuse, skip in zip(self.ups, self.fuses, skips):
            feat = up(feat)
            feat = fuse(torch.cat([feat, skip], dim=1))

        return self.head(feat)
