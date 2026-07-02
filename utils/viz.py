import torch

def save_visualization(image: torch.Tensor, gt_mask: torch.Tensor,
                       pred_mask: torch.Tensor, uncertainty: torch.Tensor,
                       out_file: str, pid: str, base_channel: int = 1):
    """
    Saves a 4-panel PNG (base image / ground truth WT overlay / predicted WT
    overlay / uncertainty heatmap) on the axial slice with the most tumor
    (ground truth WT) voxels.

    image:       (C, H, W, D)   — uses `base_channel` (default t1n=0? here t1c
                 index, see usage) as the grayscale background
    gt_mask:     (3, H, W, D)   — [WT, TC, ET] binary
    pred_mask:   (3, H, W, D)   — [WT, TC, ET] binary
    uncertainty: (3, H, W, D)   — per-channel entropy or variance map
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    image = image.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()
    uncertainty = uncertainty.cpu().numpy()

    wt_gt = gt_mask[0]
    tumor_voxels_per_slice = wt_gt.sum(axis=(0, 1))
    slice_idx = int(np.argmax(tumor_voxels_per_slice)) if tumor_voxels_per_slice.max() > 0 \
        else wt_gt.shape[-1] // 2

    bg = image[base_channel, :, :, slice_idx]
    gt_slice = gt_mask[:, :, :, slice_idx]      # (3, H, W)
    pred_slice = pred_mask[:, :, :, slice_idx]  # (3, H, W)
    unc_slice = uncertainty[:, :, :, slice_idx].mean(axis=0)  # mean over WT/TC/ET

    def overlay(ax, bg_img, mask3, title):
            ax.imshow(bg_img, cmap="gray", origin="lower")
            rgba = np.zeros((*bg_img.shape, 4))
            
            # 1. Define your three distinct RGB colors (values from 0 to 1)
            color_wt = np.array([1.0, 0.0, 0.0])  # Red
            color_tc = np.array([0.0, 1.0, 0.0])  # Green
            color_et = np.array([1.0, 0.0, 1.0])  # Magenta (distinct third color)
            
            # 2. Create boolean masks for where each region exists
            m_wt = mask3[0] > 0
            m_tc = mask3[1] > 0
            m_et = mask3[2] > 0
            
            # 3. Layer the colors sequentially. 
            # (Order matters: ET is applied last, so it sits on top of TC and WT)
            rgba[m_wt, :3] = color_wt
            rgba[m_tc, :3] = color_tc
            rgba[m_et, :3] = color_et
            
            # 4. Apply a 0.5 Alpha (transparency) only to pixels that have at least one mask
            any_mask = m_wt | m_tc | m_et
            rgba[any_mask, 3] = 0.5 
            
            ax.imshow(rgba, origin="lower")
            ax.set_title(title)
            ax.axis("off")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(bg, cmap="gray", origin="lower")
    axes[0].set_title(f"{pid}\nimage (slice {slice_idx})")
    axes[0].axis("off")

    overlay(axes[1], bg, gt_slice, "ground truth (R=WT G=TC B=ET)")
    overlay(axes[2], bg, pred_slice, "prediction (R=WT G=TC B=ET)")

    im = axes[3].imshow(unc_slice, cmap="inferno", origin="lower")
    axes[3].set_title("MC-dropout uncertainty (entropy)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
