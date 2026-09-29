"""
visualize_jepa_features.py
===========================
Sanity-check for a pretrained 4D-JEPA encoder: does it actually encode
brain anatomy, or did it just learn something degenerate?

Method (the standard "PCA-of-features" diagnostic, as used for DINO/JEPA
ViTs): run the FROZEN pretrained encoder over a full volume with
`encoder.forward(x)` (all tokens, no masking), take the resulting
per-patch token embeddings [N, D], project them down to 3 principal
components, min-max normalize those 3 components to [0,1], and paint
them as an RGB image over the patch grid. 
Usage
-----
    python visualize_jepa_features.py \
        --encoder_ckpt jepa_outputs/pretrained_encoder.pth \
        --cache_pt monai_persistent_cache/pretrain/BraTS-XXX.pt \
        --patch_size 16 --embed_dim 384 --encoder_depth 8 --encoder_heads 8 \
        --output_dir feature_viz

    # or with raw NIfTI:
    python visualize_jepa_features.py \
        --encoder_ckpt jepa_outputs/pretrained_encoder.pth \
        --t1n BraTS-XXX-t1n.nii.gz --t1c BraTS-XXX-t1c.nii.gz \
        --t2w BraTS-XXX-t2w.nii.gz --t2f BraTS-XXX-t2f.nii.gz \
        --patch_size 16 --embed_dim 384 --encoder_depth 8 --encoder_heads 8

Output
------
  {output_dir}/pca_features.png   -- grid of MRI slice vs. PCA-RGB slice
                                      at several axial/coronal/sagittal levels
  {output_dir}/pca_grid.npy       -- raw [gh, gw, gd, 3] PCA volume (for
                                      further analysis / your own plotting)
  Console: PCA explained-variance ratio of the first 3 components, and
           the foreground patch fraction used.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import VisionTransformer3D  # noqa: E402


# ─── PCA (no sklearn dependency — plain NumPy SVD) ───────────────────────────

def pca_fit_transform(X: np.ndarray, n_components: int = 3):
    """
    X: [N, D] float array.
    Returns (Y, explained_variance_ratio) where Y is [N, n_components].
    """
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # economy SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]                      # [n_components, D]
    Y = Xc @ comps.T                                # [N, n_components]
    total_var = (S ** 2).sum()
    explained = (S[:n_components] ** 2) / total_var
    return Y, explained


def minmax_norm(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ─── Volume loading ───────────────────────────────────────────────────────────

def load_from_cache_pt(path: str) -> np.ndarray:
    """Loads a build_precomputed_cache() .pt file -> [4,H,W,D] float32 numpy."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    img = data["image"]
    if isinstance(img, torch.Tensor):
        img = img.float().numpy()
    else:
        img = np.asarray(img, dtype=np.float32)
    return img


def load_from_raw_nifti(t1n, t1c, t2w, t2f, img_size) -> np.ndarray:
    import nibabel as nib

    vols = [nib.load(p).get_fdata().astype(np.float32) for p in (t1n, t1c, t2w, t2f)]
    stack = np.stack(vols, axis=0)  # [4, H, W, D]

    # Per-channel z-score over nonzero voxels only.
    for c in range(stack.shape[0]):
        ch = stack[c]
        nz = ch[ch != 0]
        if nz.size > 0:
            mean, std = nz.mean(), nz.std()
            std = std if std > 1e-8 else 1.0
            ch = np.where(ch != 0, (ch - mean) / std, 0.0)
            stack[c] = ch

    # Foreground bbox: any channel nonzero, with an 8-voxel margin (matches
    # CropForegroundd(margin=8) in train.py), shared across all 4 channels.
    fg_mask = np.any(stack != 0, axis=0)
    if fg_mask.any():
        coords = np.array(np.nonzero(fg_mask))
        mins = np.maximum(coords.min(axis=1) - 8, 0)
        maxs = np.minimum(coords.max(axis=1) + 1 + 8, fg_mask.shape)
        stack = stack[:, mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    # Pad (centered) to img_size.
    _, h, w, d = stack.shape
    H, W, D = img_size
    pad_h, pad_w, pad_d = max(H - h, 0), max(W - w, 0), max(D - d, 0)
    if pad_h or pad_w or pad_d:
        pads = [(0, 0),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (pad_d // 2, pad_d - pad_d // 2)]
        stack = np.pad(stack, pads, mode="constant", constant_values=0.0)

    # If it's larger than img_size in any dim, center-crop.
    _, h, w, d = stack.shape
    sh, sw, sd = max((h - H) // 2, 0), max((w - W) // 2, 0), max((d - D) // 2, 0)
    stack = stack[:, sh:sh + H, sw:sw + W, sd:sd + D]

    return stack.astype(np.float32)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", required=True,
                     help="Path to pretrained_encoder.pth (state_dict of VisionTransformer3D)")
    ap.add_argument("--cache_pt", default=None,
                     help="A .pt file from build_precomputed_cache (preferred input)")
    ap.add_argument("--t1n", default=None)
    ap.add_argument("--t1c", default=None)
    ap.add_argument("--t2w", default=None)
    ap.add_argument("--t2f", default=None)
    ap.add_argument("--img_size", type=int, nargs=3, default=(128, 128, 128))
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--encoder_depth", type=int, default=8)
    ap.add_argument("--encoder_heads", type=int, default=8)
    ap.add_argument("--fg_threshold", type=float, default=1e-6,
                     help="A patch is 'foreground' if the mean |intensity| "
                          "of its voxels (summed across the 4 modalities) "
                          "exceeds this. Raw z-scored brain tissue is never "
                          "exactly 0, true background is.")
    ap.add_argument("--slices", type=int, nargs="+", default=None,
                     help="Which grid-index slices (along each of the 3 axes) "
                          "to plot. Defaults to 5 evenly spaced slices per axis.")
    ap.add_argument("--output_dir", default="feature_viz")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Load the volume -------------------------------------------------
    if args.cache_pt:
        volume = load_from_cache_pt(args.cache_pt)   # [4,H,W,D]
    else:
        missing = [n for n, v in [("t1n", args.t1n), ("t1c", args.t1c),
                                   ("t2w", args.t2w), ("t2f", args.t2f)] if v is None]
        if missing:
            raise ValueError(
                f"Provide either --cache_pt, or all four of --t1n/--t1c/--t2w/--t2f "
                f"(missing: {missing})"
            )
        volume = load_from_raw_nifti(args.t1n, args.t1c, args.t2w, args.t2f, args.img_size)

    print(f"Volume shape: {volume.shape}")
    x = torch.from_numpy(volume).float().unsqueeze(0).to(device)  # [1,4,H,W,D]

    # ---- 2. Build encoder + load weights ------------------------------------
    encoder = VisionTransformer3D(
        img_size=tuple(volume.shape[1:]),
        patch_size=args.patch_size,
        in_channels=4,
        embed_dim=args.embed_dim,
        depth=args.encoder_depth,
        num_heads=args.encoder_heads,
    ).to(device)

    state = torch.load(args.encoder_ckpt, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warning] load_state_dict: missing={missing}, unexpected={unexpected}")
    encoder.eval()

    # ---- 3. Forward pass: full-volume tokens, no masking --------------------
    with torch.no_grad():
        tokens = encoder(x)          # [1, N, D] -- uses forward(), i.e. ALL tokens
    tokens_np = tokens[0].cpu().numpy()   # [N, D]

    gh, gw, gd = encoder.patch_embed.grid_size
    N = gh * gw * gd
    assert tokens_np.shape[0] == N

    # ---- 4. Foreground mask from the INPUT (true zero-background) -----------
    P = args.patch_size
    # mean abs intensity per patch, summed over the 4 channels
    vol_t = x[0].cpu().numpy()  # [4,H,W,D]
    patch_mean_abs = np.zeros(N, dtype=np.float32)
    idx = 0
    for hi in range(gh):
        for wi in range(gw):
            for di in range(gd):
                block = vol_t[:, hi*P:(hi+1)*P, wi*P:(wi+1)*P, di*P:(di+1)*P]
                patch_mean_abs[idx] = np.abs(block).mean()
                idx += 1
    fg_mask = patch_mean_abs > args.fg_threshold
    fg_frac = fg_mask.mean()
    print(f"Foreground patch fraction: {fg_frac:.3f}  ({fg_mask.sum()}/{N} patches)")
    if fg_mask.sum() < 10:
        print("  [warning] Very few foreground patches detected -- check "
              "--fg_threshold or that the volume isn't blank.")

    # ---- 5. PCA fit on foreground tokens only, applied to all tokens --------
    fg_tokens = tokens_np[fg_mask]
    mean = fg_tokens.mean(axis=0, keepdims=True)
    _, explained = pca_fit_transform(fg_tokens, n_components=3)
    # Refit to get the projection basis explicitly so we can apply it to ALL tokens
    Xc = fg_tokens - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:3]                                  # [3, D]

    proj_all = (tokens_np - mean) @ comps.T          # [N, 3], for every patch
    proj_fg = proj_all[fg_mask]

    print(f"PCA explained variance ratio (top 3, computed on foreground tokens): "
          f"{explained.round(3).tolist()}")

    # Normalize each of the 3 channels using ONLY foreground statistics,
    # so the background doesn't skew the color scale.
    rgb = np.zeros((N, 3), dtype=np.float32)
    for c in range(3):
        lo, hi = proj_fg[:, c].min(), proj_fg[:, c].max()
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        rgb[:, c] = np.clip((proj_all[:, c] - lo) / (hi - lo), 0, 1)
    rgb[~fg_mask] = 0.0   # paint background black

    pca_grid = rgb.reshape(gh, gw, gd, 3)
    np.save(output_dir / "pca_grid.npy", pca_grid)

    # ---- 6. Upsample the (coarse) PCA grid back to voxel resolution ---------
    pca_grid_t = torch.from_numpy(pca_grid).permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,gh,gw,gd]
    H, W, D = volume.shape[1:]
    pca_full = torch.nn.functional.interpolate(
        pca_grid_t, size=(H, W, D), mode="nearest"
    )[0].permute(1, 2, 3, 0).numpy()   # [H,W,D,3]

    # ---- 7. Plot: MRI slice next to PCA-RGB slice, 3 axes --------------------
    t1c_vol = volume[1]  # index 1 == t1c, per the (t1n,t1c,t2w,t2f) channel order

    def norm_gray(sl):
        lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
        return np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)

    axes_info = [
        ("Axial (H)", lambda v, i: v[i, :, :], H),
        ("Coronal (W)", lambda v, i: v[:, i, :], W),
        ("Sagittal (D)", lambda v, i: v[:, :, i], D),
    ]
    n_slices = 5
    fig, axarr = plt.subplots(3 * 2, n_slices, figsize=(3 * n_slices, 12))

    for row_axis, (axis_name, slicer, extent) in enumerate(axes_info):
        slice_idx = args.slices if args.slices else list(
            np.linspace(int(extent * 0.15), int(extent * 0.85), n_slices).astype(int)
        )
        for col, si in enumerate(slice_idx[:n_slices]):
            gray_slice = norm_gray(slicer(t1c_vol, si))
            if axis_name == "Axial (H)":
                rgb_slice = pca_full[si, :, :, :]
            elif axis_name == "Coronal (W)":
                rgb_slice = pca_full[:, si, :, :]
            else:
                rgb_slice = pca_full[:, :, si, :]

            ax_top = axarr[row_axis * 2, col]
            ax_bot = axarr[row_axis * 2 + 1, col]
            ax_top.imshow(gray_slice.T, cmap="gray", origin="lower")
            ax_top.set_title(f"{axis_name} idx={si}\nT1c", fontsize=8)
            ax_top.axis("off")
            ax_bot.imshow(np.transpose(rgb_slice, (1, 0, 2)), origin="lower")
            ax_bot.set_title("PCA(features)", fontsize=8)
            ax_bot.axis("off")

    plt.tight_layout()
    out_path = output_dir / "pca_features.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization -> {out_path}")
    print(f"Saved raw PCA grid   -> {output_dir / 'pca_grid.npy'}")


if __name__ == "__main__":
    main()
