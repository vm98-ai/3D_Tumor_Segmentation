"""
4D-JEPA: Pretraining + Fine-tuning for BraTS 2024
===================================================
"""

import os
import json
import argparse
import math
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import MultiBlock3DMasker
from model import (
    VisionTransformer3D,
    EMATargetEncoder,
    SpatialPredictor3D,
    CrossModalPredictor,
    UNETRDecoder3D,
)


# ────────────────────────────────────────────────────────────────────────────
# JEPA Loss
# ────────────────────────────────────────────────────────────────────────────

# import torch
# import torch.nn.functional as F


# def fisher_geometry_loss(
#     predictions,
#     targets,
#     num_directions=4,
#     eps=1e-3,
# ):
#     """
#     Local Fisher / information-geometric consistency loss.

#     Supports:
#         [B, Nt, D]
#         [N, D]

#     The loss compares the local metric induced by the
#     prediction and target representations.
#     """

#     total_loss = 0.0
#     num_levels = 0

#     for pred, tgt in zip(predictions, targets):

#         # ---------------------------------------------------------
#         # Normalize representations
#         # ---------------------------------------------------------
#         pred = F.normalize(pred, dim=-1)
#         tgt = F.normalize(tgt.detach(), dim=-1)

#         # ---------------------------------------------------------
#         # Accept both:
#         #
#         # [B, Nt, D]
#         # [N, D]
#         # ---------------------------------------------------------
#         if pred.dim() == 2:
#             pred = pred.unsqueeze(1)
#             tgt = tgt.unsqueeze(1)

#         elif pred.dim() != 3:
#             raise ValueError(
#                 f"Expected representation with 2 or 3 dimensions, "
#                 f"got shape {pred.shape}"
#             )

#         B, Nt, D = pred.shape

#         # ---------------------------------------------------------
#         # Random tangent directions
#         # ---------------------------------------------------------
#         directions = torch.randn(
#             B,
#             Nt,
#             num_directions,
#             D,
#             device=pred.device,
#             dtype=pred.dtype,
#         )

#         directions = F.normalize(
#             directions,
#             dim=-1,
#         )

#         # ---------------------------------------------------------
#         # Expand representations
#         # ---------------------------------------------------------
#         pred = pred.unsqueeze(2)
#         tgt = tgt.unsqueeze(2)

#         # ---------------------------------------------------------
#         # Local perturbations
#         # ---------------------------------------------------------
#         pred_plus = F.normalize(
#             pred + eps * directions,
#             dim=-1,
#         )

#         pred_minus = F.normalize(
#             pred - eps * directions,
#             dim=-1,
#         )

#         tgt_plus = F.normalize(
#             tgt + eps * directions,
#             dim=-1,
#         )

#         tgt_minus = F.normalize(
#             tgt - eps * directions,
#             dim=-1,
#         )

#         # ---------------------------------------------------------
#         # Finite-difference directional derivatives
#         # ---------------------------------------------------------
#         pred_delta = (
#             pred_plus - pred_minus
#         ) / (2.0 * eps)

#         tgt_delta = (
#             tgt_plus - tgt_minus
#         ) / (2.0 * eps)

#         # ---------------------------------------------------------
#         # Local metric:
#         #
#         # v^T F v ≈ ||Jv||²
#         # ---------------------------------------------------------
#         pred_fisher = pred_delta.pow(2).sum(dim=-1)
#         tgt_fisher = tgt_delta.pow(2).sum(dim=-1)

#         # ---------------------------------------------------------
#         # Compare local information geometry
#         # ---------------------------------------------------------
#         loss = F.mse_loss(
#             pred_fisher,
#             tgt_fisher.detach(),
#             reduction="mean",
#         )

#         total_loss = total_loss + loss
#         num_levels += 1

#     return total_loss / max(num_levels, 1)

# def jepa_fisher_loss(
#     predictions,
#     targets,
#     lambda_fisher=0.1,
#     num_directions=4,
#     eps=1e-3,
# ):
#     # Standard JEPA loss
#     jepa = 0.0
#     count = 0

#     for pred, tgt in zip(predictions, targets):

#         pred_n = F.normalize(pred, dim=-1)
#         tgt_n = F.normalize(tgt.detach(), dim=-1)

#         jepa += F.mse_loss(
#             pred_n,
#             tgt_n,
#             reduction="sum"
#         )

#         count += pred_n.shape[0] * pred_n.shape[1]

#     jepa = jepa / max(count, 1)

#     # Fisher geometry loss
#     fisher = fisher_geometry_loss(
#         predictions,
#         targets,
#         num_directions=num_directions,
#         eps=eps,
#     )

#     return jepa + lambda_fisher * fisher

def jepa_loss(
    predictions: List[torch.Tensor],   # list of B × [Nt, D]
    targets: List[torch.Tensor],        # list of B × [Nt, D]
) -> torch.Tensor:
    total, count = 0.0, 0
    for pred, tgt in zip(predictions, targets):
        pred_n = F.normalize(pred, dim=-1)
        tgt_n = F.normalize(tgt.detach(), dim=-1)
        total = total + F.mse_loss(pred_n, tgt_n, reduction="sum")
        count += pred_n.shape[0] * pred_n.shape[1]   # tokens × dims
    return total / max(count, 1)


# ────────────────────────────────────────────────────────────────────────────
# Phase 1: Pretraining
# ────────────────────────────────────────────────────────────────────────────

class JEPA4DPretrainer(nn.Module):

    def __init__(
        self,
        img_size: tuple = (128, 128, 128),
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        tau_start: float = 0.996,
        tau_end: float = 1.0,
        total_steps: int = 100_000,
        mask_ratio: float = 0.65,
        num_mask_blocks: int = 6,
        use_cross_modal: bool = True,
        cross_modal_weight: float = 0.25,
    ):
        super().__init__()
        self.use_cross_modal = use_cross_modal
        self.cross_modal_weight = cross_modal_weight

        self.encoder = VisionTransformer3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=4,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )
        grid = self.encoder.patch_embed.grid_size

        self.target_encoder = EMATargetEncoder(
            self.encoder, tau_start=tau_start, tau_end=tau_end,
            total_steps=total_steps,
        )

        self.predictor = SpatialPredictor3D(
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            num_heads=predictor_heads,
            depth=predictor_depth,
            grid_size=grid,
        )

        if use_cross_modal:
            self.cross_modal_pred = CrossModalPredictor(
                embed_dim=embed_dim,
                predictor_dim=predictor_dim // 2,
                num_heads=predictor_heads,
            )

        self.masker = MultiBlock3DMasker(
            grid_size=grid,
            mask_ratio=mask_ratio,
            num_blocks=num_mask_blocks,
        )

        self.num_patches = self.encoder.patch_embed.num_patches

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, 4, H, W, D]  — channel order assumed (t1n, t1c, t2w, t2f)
               to match the data_dicts built in `pretrain()` below.
        Returns:
            dict with 'loss', 'spatial_loss', 'cross_modal_loss'
        """
        B, C, H, W, D = x.shape
        device = x.device

        # 1. Sample masks
        context_idx, target_idx = self.masker.sample(B, device)

        # 2. Context encoder — only visible tokens
        ctx_repr = self.encoder.forward_subset(x, context_idx)  # list B × [Nc, D]

        # 3. Target encoder — all tokens (no grad)
        with torch.no_grad():
            all_tgt_repr = self.target_encoder(x)               # [B, N, D]

        tgt_repr = [all_tgt_repr[b][target_idx[b]] for b in range(B)]

        # 4. Predictor: context repr → predicted target repr
        pred_repr = self.predictor(
            ctx_repr, context_idx, target_idx, self.num_patches
        )

        # 5. Spatial JEPA loss
        spatial_loss = jepa_loss(pred_repr, tgt_repr)
        total_loss = spatial_loss

        cross_loss = torch.tensor(0.0, device=device)
        if self.use_cross_modal:
            # T1c (channel 1) repr → predict T2f (channel 3) repr, both via
            # the shared single-channel projection (forward_single_modality).
            x_t1c = x[:, 1:2]   # [B, 1, H, W, D]
            x_t2f = x[:, 3:4]   # [B, 1, H, W, D]

            with torch.no_grad():
                t2f_repr = self.target_encoder.encoder.forward_single_modality(x_t2f)  # [B, N, D]

            t1c_repr = self.encoder.forward_single_modality(x_t1c)  # [B, N, D]

            pos_emb = self.encoder.pos_embed.pos_embed.expand(B, -1, -1)  # [B, N, D]
            pred_cross = self.cross_modal_pred(t1c_repr, pos_emb)          # [B, N, D]

            cross_loss = F.mse_loss(
                F.normalize(pred_cross, dim=-1),
                F.normalize(t2f_repr.detach(), dim=-1),
            )
            total_loss = spatial_loss + self.cross_modal_weight * cross_loss

        return {
            "loss": total_loss,
            "spatial_loss": spatial_loss.item(),
            "cross_modal_loss": cross_loss.item(),
        }


def build_precomputed_cache(
    data_dicts: list,
    deterministic_transform,
    cache_dir: str,
    force: bool = False,
) -> "Path":
    import shutil
    from tqdm.auto import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_written, n_skipped = 0, 0
    warned_space = False
    pbar = tqdm(data_dicts, desc=f"Precomputing -> {cache_dir}", unit="patient")
    for item in pbar:
        image_paths = item["image"]
        patient_id = Path(image_paths[0] if isinstance(image_paths, list) else image_paths).parent.name
        out_path = cache_dir / f"{patient_id}.pt"

        if out_path.exists() and not force:
            n_skipped += 1
            continue

        processed = deterministic_transform(dict(item))
        tmp_path = out_path.with_suffix(".pt.tmp")
        try:
            torch.save(processed, tmp_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            free_gb = shutil.disk_usage(cache_dir).free / 1e9
            raise RuntimeError(
                f"Failed writing cache file for patient '{patient_id}' "
                f"({free_gb:.1f} GB free on {cache_dir}'s volume)."
            ) from e
        tmp_path.replace(out_path)   # atomic-ish: avoids half-written files on interrupt
        n_written += 1

        if not warned_space and n_written == 1:
            item_bytes = out_path.stat().st_size
            remaining = len(data_dicts) - n_written - n_skipped
            est_remaining_bytes = item_bytes * remaining
            free_bytes = shutil.disk_usage(cache_dir).free
            pbar.write(
                f"  ~{item_bytes / 1e6:.0f} MB/patient -> est. "
                f"{est_remaining_bytes / 1e9:.1f} GB more needed, "
                f"{free_bytes / 1e9:.1f} GB free on this volume."
            )
            if est_remaining_bytes > free_bytes:
                pbar.write(
                    "  WARNING: estimated cache size exceeds free disk space. "
                    "Consider a bigger/other --cache_dir, or Ctrl+C now."
                )
            warned_space = True

    print(f"  Precompute cache: {n_written} written, {n_skipped} already present -> {cache_dir}")
    return cache_dir


class PrecomputedCacheDataset(torch.utils.data.Dataset):

    def __init__(self, cache_dir: str, random_transform=None):
        self.paths = sorted(Path(cache_dir).glob("*.pt"))
        if not self.paths:
            raise RuntimeError(
                f"No precomputed cache files (*.pt) found in {cache_dir} — "
                "did build_precomputed_cache() run first?"
            )
        self.random_transform = random_transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        data = torch.load(self.paths[idx], map_location="cpu", weights_only=False)
        if self.random_transform is not None:
            data = self.random_transform(data)
        return data


def pretrain(
    data_root: str,
    output_dir: str = "jepa_outputs",
    epochs: int = 200,
    batch_size: int = 2,
    lr: float = 1.5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 20,
    patch_size: int = 16,
    embed_dim: int = 384,     # smaller default to fit on 1 GPU
    encoder_depth: int = 8,
    encoder_heads: int = 8,
    num_workers: int = 4,
    img_size: tuple = (128, 128, 128),
    use_cross_modal: bool = True,
    resume: Optional[str] = None,
    cache_rate: float = 0.0,
    cache_dir: str = "monai_persistent_cache/pretrain",
    cache_mode: str = "precomputed",   # "precomputed" | "persistent" | "inmemory"
    force_recache: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from sklearn.model_selection import train_test_split
    import nibabel as nib
    from tqdm.auto import tqdm

    data_dir = data_root
    patient_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    data_dicts = []
    for p_id in patient_folders:
        patient_path = os.path.join(data_dir, p_id)
        data_dicts.append({
            # channel order: t1n(0), t1c(1), t2w(2), t2f(3)
            "image": [
                os.path.join(patient_path, f"{p_id}-t1n.nii.gz"),
                os.path.join(patient_path, f"{p_id}-t1c.nii.gz"),
                os.path.join(patient_path, f"{p_id}-t2w.nii.gz"),
                os.path.join(patient_path, f"{p_id}-t2f.nii.gz")
            ]
        })

    def clean_pretrain_data_dicts(data_dicts):
        print("Scanning dataset for shape mismatches (Pretraining Mode)...")
        clean_dicts = []
        bad_patients = []

        for patient in tqdm(data_dicts):
            image_paths = patient["image"]
            shapes = []
            for path in image_paths:
                try:
                    img = nib.load(path)
                    shapes.append(img.shape)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    shapes.append(None)

            if all(shape == shapes[0] for shape in shapes) and None not in shapes:
                clean_dicts.append(patient)
            else:
                bad_patients.append((patient["image"][0], f"Modality mismatch: {shapes}"))

        print(f"\nScan complete. Found {len(bad_patients)} corrupted/mismatched patients.")
        if bad_patients:
            print("Examples of bad data:")
            for bad in bad_patients[:5]:
                print(f" - {bad[0]}: {bad[1]}")

        return clean_dicts

    validated_data_dicts = clean_pretrain_data_dicts(data_dicts)
    train_files, val_files = train_test_split(validated_data_dicts, test_size=0.01, random_state=42)
    print(f"Proceeding with {len(train_files)} training and {len(val_files)} validation patients.")

    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        NormalizeIntensityd,
        EnsureTyped,
        SpatialPadd,
        RandSpatialCropd,
        CropForegroundd,
    )

    train_transforms = Compose([
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image", margin=8, allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=(128, 128, 128), mode="constant", value=0),
        # float16 halves the on-disk/cache footprint vs float32.
        EnsureTyped(keys=["image"], dtype=torch.float16, track_meta=False),
    ])
    # Everything below only runs at train time, never cached.
    random_transforms = Compose([
        RandSpatialCropd(keys=["image"], roi_size=(128, 128, 128), random_size=False),
        EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),  # undo float16 for training math
    ])

    from monai.data import CacheDataset, PersistentDataset, DataLoader
    from monai.data.utils import pickle_hashing

    if cache_mode == "inmemory":
        train_ds = CacheDataset(
            data=train_files,
            transform=Compose(list(train_transforms.transforms) + list(random_transforms.transforms)),
            cache_rate=cache_rate, num_workers=num_workers,
        )
    elif cache_mode == "persistent":
        base_ds = PersistentDataset(
            data=train_files, transform=train_transforms, cache_dir=cache_dir,
            hash_transform=pickle_hashing,
        )
        from monai.data import Dataset as MonaiDataset
        train_ds = MonaiDataset(data=base_ds, transform=random_transforms)  # type: ignore
    else:  # "precomputed"
        cache_root = build_precomputed_cache(
            train_files, train_transforms, cache_dir, force=force_recache,
        )
        train_ds = PrecomputedCacheDataset(cache_root, random_transform=random_transforms)

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch

    model = JEPA4DPretrainer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        encoder_heads=encoder_heads,
        predictor_dim=embed_dim // 2,
        total_steps=total_steps,
        use_cross_modal=use_cross_modal,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_epochs * steps_per_epoch:
            return step / max(1, warmup_epochs * steps_per_epoch)
        progress = (step - warmup_epochs * steps_per_epoch) / max(
            1, total_steps - warmup_epochs * steps_per_epoch
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    history = []
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = {"loss": 0, "spatial_loss": 0, "cross_modal_loss": 0}

        pbar = tqdm(
            enumerate(loader), total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{epochs}", unit="step", leave=False,
        )
        for step, batch in pbar:
            images = batch["image"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(images)
                loss = out["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            model.target_encoder.update(model.encoder)

            for k in epoch_losses:
                epoch_losses[k] += out[k] if isinstance(out[k], float) else out[k].item()
            global_step += 1

            lr_now = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{epoch_losses['loss']/(step+1):.4f}",
                spatial=f"{epoch_losses['spatial_loss']/(step+1):.4f}",
                cross=f"{epoch_losses['cross_modal_loss']/(step+1):.4f}",
                lr=f"{lr_now:.2e}",
            )

            del images, out, loss

        pbar.close()
        n = len(loader)
        entry = {k: v / n for k, v in epoch_losses.items()}
        entry["epoch"] = epoch + 1
        history.append(entry)

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "encoder": model.encoder.state_dict(),
        }, str(f"{output_dir}/latest_pretrain.pth"))

        with open(f"{output_dir}/pretrain_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"  \u2713 Epoch {epoch+1} | "
            f"avg loss={entry['loss']:.4f}  "
            f"spatial={entry['spatial_loss']:.4f}  "
            f"cross={entry['cross_modal_loss']:.4f}"
        )

    torch.save(
        model.encoder.state_dict(),
        str(f"{output_dir}/pretrained_encoder.pth"),
    )
    print(f"\nPretrained encoder saved to {output_dir}/pretrained_encoder.pth")


# ────────────────────────────────────────────────────────────────────────────
# Phase 2: Fine-tuning — attach a UNETR-style decoder to the (pre)trained encoder
# ────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss averaged over all foreground classes, with deep-supervision support."""

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5,
                 include_background: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def _dice_per_class(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C = pred.shape[:2]
        pred_soft = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=C).permute(0, 4, 1, 2, 3).float()

        start_cls = 0 if self.include_background else 1
        dice_scores = []
        for c in range(start_cls, C):
            p = pred_soft[:, c].contiguous().view(B, -1)
            t = target_one_hot[:, c].contiguous().view(B, -1)
            inter = (p * t).sum(dim=1)
            union = p.sum(dim=1) + t.sum(dim=1)
            dice = (2 * inter + self.smooth) / (union + self.smooth)
            dice_scores.append(dice.mean())

        return torch.stack(dice_scores).mean()

    def forward(self, pred, target: torch.Tensor) -> torch.Tensor:
        if isinstance(pred, list):
            weights = [0.5 ** (len(pred) - 1 - i) for i in range(len(pred))]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            loss = torch.tensor(0.0, device=target.device)
            for p, w in zip(pred, weights):
                if p.shape[2:] != target.shape[1:]:
                    t_ds = F.interpolate(
                        target.float().unsqueeze(1), size=p.shape[2:], mode="nearest",
                    ).squeeze(1).long()
                else:
                    t_ds = target
                loss = loss + w * (1 - self._dice_per_class(p, t_ds))
            return loss
        else:
            return 1 - self._dice_per_class(pred, target)


class CombinedLoss(nn.Module):
    """Dice + Cross-Entropy, with deep supervision support."""

    def __init__(self, num_classes: int = 4, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(num_classes, include_background=False)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target: torch.Tensor) -> torch.Tensor:
        finest = pred[-1] if isinstance(pred, list) else pred
        ce_loss = F.cross_entropy(finest, target.squeeze(1).long())
        dice_loss = self.dice(pred, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class JEPA4DFineTuner(nn.Module):

    def __init__(self, encoder: VisionTransformer3D, num_classes: int = 4,
                 freeze_encoder: bool = False, freeze_first_n_blocks: int = 8,
                 in_channels: int = 4):
        super().__init__()
        self.encoder = encoder
        depth = len(encoder.blocks)
        patch_size = encoder.patch_embed.patch_size
        n_stages = int(math.log2(patch_size))

        fracs = [(i + 1) / n_stages for i in range(n_stages - 1)] + [1.0]
        indices = sorted(set(max(1, round(depth * f)) for f in fracs))
        while len(indices) < n_stages:
            missing = n_stages - len(indices)
            candidates = [i for i in range(1, depth + 1) if i not in indices]
            indices = sorted(indices + candidates[:missing])
        if indices[-1] != depth:
            indices[-1] = depth
        self.layer_indices = indices
        print(f"UNETR decoder skip layers (block indices, 1-indexed): {self.layer_indices}")

        self.decoder = UNETRDecoder3D(
            embed_dim=encoder.embed_dim,
            grid_size=encoder.patch_embed.grid_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
        )

        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad_(False)
        elif freeze_first_n_blocks > 0:
            for p in encoder.patch_embed.parameters():
                p.requires_grad_(False)
            for i, block in enumerate(encoder.blocks):
                if i < freeze_first_n_blocks:
                    for p in block.parameters():
                        p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encoder.forward_with_hidden_states(x, self.layer_indices)
        return self.decoder(hidden_states, x, self.layer_indices)


def compute_dice_per_class(logits, labels, num_classes: int = 4, smooth: float = 1e-6) -> torch.Tensor:
    """
    Per-class Dice from raw logits and integer label map.
    BraTS labels: 0=background, 1=NCR, 2=ED, 3=ET.
    """
    preds = torch.argmax(logits, dim=1)
    dice_scores = torch.zeros(num_classes, device=logits.device)
    for c in range(num_classes):
        pred_c = (preds == c).float()
        true_c = (labels == c).float()
        intersection = (pred_c * true_c).sum()
        dice_scores[c] = (2.0 * intersection + smooth) / (pred_c.sum() + true_c.sum() + smooth)
    return dice_scores


def compute_brats_region_dice(logits, labels, smooth: float = 1e-6) -> dict:
    """WT = whole tumour (1,2,3), TC = tumour core (1,3), ET = enhancing tumour (3)."""
    preds = torch.argmax(logits, dim=1)

    def _dice(pred_mask, true_mask):
        inter = (pred_mask & true_mask).float().sum()
        union = pred_mask.float().sum() + true_mask.float().sum()
        return ((2.0 * inter + smooth) / (union + smooth)).item()

    wt_pred = preds >= 1; wt_true = labels >= 1
    tc_pred = (preds == 1) | (preds == 3)
    tc_true = (labels == 1) | (labels == 3)
    et_pred = preds == 3; et_true = labels == 3

    return {
        "dice_wt": _dice(wt_pred, wt_true),
        "dice_tc": _dice(tc_pred, tc_true),
        "dice_et": _dice(et_pred, et_true),
    }


def finetune(
    data_root: str,
    pretrained_encoder: str,
    output_dir: str = "jepa_finetune",
    epochs: int = 150,
    batch_size: int = 2,
    lr: float = 5e-5,
    patch_size: int = 16,
    embed_dim: int = 384,
    encoder_depth: int = 8,
    encoder_heads: int = 8,
    img_size: tuple = (128, 128, 128),
    freeze_first_n: int = 6,
    num_workers: int = 4,
    cache_rate: float = 0.0,
    train_cache_dir: str = "monai_persistent_cache/finetune_train",
    val_cache_dir: str = "monai_persistent_cache/finetune_val",
    num_samples_per_crop: int = 1,
    cache_mode: str = "precomputed",   # "precomputed" | "persistent" | "inmemory"
    force_recache: bool = False,
    resume: Optional[str] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from sklearn.model_selection import train_test_split
    import nibabel as nib
    from tqdm.auto import tqdm

    data_dir = data_root
    patient_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    data_dicts = []
    for p_id in patient_folders:
        patient_path = os.path.join(data_dir, p_id)
        data_dicts.append({
            "image": [
                os.path.join(patient_path, f"{p_id}-t1n.nii.gz"),
                os.path.join(patient_path, f"{p_id}-t1c.nii.gz"),
                os.path.join(patient_path, f"{p_id}-t2w.nii.gz"),
                os.path.join(patient_path, f"{p_id}-t2f.nii.gz"),
            ],
            "mask": os.path.join(patient_path, f"{p_id}-seg.nii.gz"),
        })

    def clean_data_dicts(data_dicts):
        print("Scanning dataset for shape mismatches...")
        clean_dicts, bad_patients = [], []
        for patient in tqdm(data_dicts):
            shapes = []
            for path in patient["image"]:
                try:
                    shapes.append(nib.load(path).shape)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    shapes.append(None)
            if None not in shapes and all(s == shapes[0] for s in shapes):
                try:
                    mask_shape = nib.load(patient["mask"]).shape
                except Exception:
                    bad_patients.append((patient["image"][0], "Mask load error"))
                    continue
                if mask_shape == shapes[0]:
                    clean_dicts.append(patient)
                else:
                    bad_patients.append((patient["image"][0], "Mask shape mismatch"))
            else:
                bad_patients.append((patient["image"][0], f"Modality mismatch: {shapes}"))
        print(f"Scan complete. {len(bad_patients)} bad patients removed.")
        return clean_dicts

    validated_data_dicts = clean_data_dicts(data_dicts)
    train_files, val_files = train_test_split(validated_data_dicts, test_size=0.2, random_state=42)
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        NormalizeIntensityd, RandCropByPosNegLabeld,
        EnsureTyped, SpatialPadd,
        RandAffined, Rand3DElasticd, RandGaussianNoised,
        RandAdjustContrastd, RandBiasFieldd,
        CropForegroundd,
    )
    from monai.data import CacheDataset, PersistentDataset, DataLoader as MonaiDataLoader
    from monai.data.utils import pickle_hashing

    det_train_transforms = Compose([
        LoadImaged(keys=["image", "mask"], image_only=True, ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "mask"]),
        CropForegroundd(keys=["image", "mask"], source_key="image", margin=8, allow_smaller=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "mask"], spatial_size=(128, 128, 128), mode="constant", value=0),
        EnsureTyped(keys="image", dtype=torch.float16, track_meta=False),
        EnsureTyped(keys="mask", dtype=torch.uint8, track_meta=False),
    ])
    rand_train_transforms = Compose([
        EnsureTyped(keys=["image", "mask"], dtype=torch.float32, track_meta=False),
        RandGaussianNoised(keys="image", prob=0.1, std=0.1),
        RandAdjustContrastd(keys="image", prob=0.1, gamma=(0.5, 2.0)),
        RandBiasFieldd(keys="image", prob=0.2),
        RandAffined(
            keys=["image", "mask"], prob=0.3,
            rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
            padding_mode="zeros",
            mode=("bilinear", "nearest"),
        ),
        Rand3DElasticd(
            keys=["image", "mask"], prob=0.2,
            sigma_range=(5, 7), magnitude_range=(50, 100),
            mode=("bilinear", "nearest"),
        ),
        RandCropByPosNegLabeld(
            keys=["image", "mask"], label_key="mask",
            spatial_size=(128, 128, 128), pos=2, neg=1,
            num_samples=num_samples_per_crop,
            image_key="image", image_threshold=0,
        ),
        EnsureTyped(keys=["image", "mask"], dtype=torch.float32, track_meta=False),
    ])

    det_val_transforms = Compose([
        LoadImaged(keys=["image", "mask"], image_only=True, ensure_channel_first=False),
        EnsureChannelFirstd(keys=["mask"]),
        CropForegroundd(keys=["image", "mask"], source_key="image", margin=8, allow_smaller=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image", "mask"], spatial_size=(128, 128, 128), mode="constant", value=0),
        EnsureTyped(keys="image", dtype=torch.float16, track_meta=False),
        EnsureTyped(keys="mask", dtype=torch.uint8, track_meta=False),
    ])
    val_cast_transform = Compose([
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=False),
    ])

    if cache_mode == "inmemory":
        train_ds = CacheDataset(
            data=train_files,
            transform=Compose(list(det_train_transforms.transforms) + list(rand_train_transforms.transforms)),
            cache_rate=cache_rate, num_workers=num_workers,
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=Compose(list(det_val_transforms.transforms) + list(val_cast_transform.transforms)),
            cache_rate=cache_rate, num_workers=num_workers,
        )
    elif cache_mode == "persistent":
        from monai.data import Dataset as MonaiDataset
        base_train_ds = PersistentDataset(data=train_files, transform=det_train_transforms,
                                           cache_dir=train_cache_dir, hash_transform=pickle_hashing)
        train_ds = MonaiDataset(data=base_train_ds, transform=rand_train_transforms)  # type: ignore
        base_val_ds = PersistentDataset(data=val_files, transform=det_val_transforms,
                                         cache_dir=val_cache_dir, hash_transform=pickle_hashing)
        val_ds = MonaiDataset(data=base_val_ds, transform=val_cast_transform)  # type: ignore
    else:  # "precomputed"
        train_cache_root = build_precomputed_cache(
            train_files, det_train_transforms, train_cache_dir, force=force_recache,
        )
        val_cache_root = build_precomputed_cache(
            val_files, det_val_transforms, val_cache_dir, force=force_recache,
        )
        train_ds = PrecomputedCacheDataset(train_cache_root, random_transform=rand_train_transforms)
        val_ds = PrecomputedCacheDataset(val_cache_root, random_transform=val_cast_transform)

    train_loader = MonaiDataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = MonaiDataLoader(
        val_ds, batch_size=1, num_workers=max(1, num_workers // 2),
        pin_memory=(device.type == "cuda"),
    )

    encoder = VisionTransformer3D(
        img_size=img_size, patch_size=patch_size,
        embed_dim=embed_dim, depth=encoder_depth, num_heads=encoder_heads,
    )
    ckpt = torch.load(pretrained_encoder, map_location="cpu")
    encoder.load_state_dict(ckpt)
    print(f"Loaded pretrained encoder from {pretrained_encoder}")

    model = JEPA4DFineTuner(encoder, num_classes=4, freeze_first_n_blocks=freeze_first_n).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (fine-tune): {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    criterion = CombinedLoss(num_classes=4, dice_weight=0.5, ce_weight=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_mean_dice = 0.0
    start_epoch = 0

    # ── Resume from a finetune checkpoint (e.g. latest_finetune.pth) ──────
    if resume:
        print(f"Resuming fine-tuning from checkpoint: {resume}")
        resume_ckpt = torch.load(resume, map_location="cpu")

        model.load_state_dict(resume_ckpt["model"])
        model.to(device)

        if "optimizer" in resume_ckpt and resume_ckpt["optimizer"] is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            # Make sure any restored optimizer state tensors live on the
            # right device (map_location="cpu" above puts them on CPU).
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        if "scheduler" in resume_ckpt and resume_ckpt["scheduler"] is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler"])

        if "scaler" in resume_ckpt and resume_ckpt["scaler"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])

        start_epoch = resume_ckpt.get("epoch", 0)
        best_mean_dice = resume_ckpt.get("best_mean_dice", 0.0)

        # Reload history so finetune_history.json isn't truncated on resume.
        history_path = output_dir / "finetune_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            # Trim any entries beyond start_epoch, in case the run crashed
            # mid-write or history.json is ahead of the checkpoint for
            # some other reason.
            history = [h for h in history if h.get("epoch", 0) <= start_epoch]

        print(
            f"Resumed at epoch {start_epoch}/{epochs} "
            f"(best_mean_dice so far = {best_mean_dice:.4f}, "
            f"{len(history)} history entries loaded)"
        )

        if start_epoch >= epochs:
            print(
                f"Checkpoint epoch ({start_epoch}) >= requested --epochs ({epochs}); "
                "nothing to do. Increase --epochs to continue training."
            )
            return
    # ────────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_sum = 0.0
        train_pbar = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs} [train]", unit="step", leave=False,
        )
        for step, batch in train_pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["mask"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item()

            train_pbar.set_postfix(loss=f"{train_loss_sum/(step+1):.4f}")

            del images, labels, logits, loss

        train_pbar.close()
        avg_train_loss = train_loss_sum / len(train_loader)
        scheduler.step()

        val_loss_sum, dice_wt_sum, dice_tc_sum, dice_et_sum, n_val = 0.0, 0.0, 0.0, 0.0, 0
        per_class_sum = torch.zeros(4)

        model.eval()
        val_pbar = tqdm(val_loader, total=len(val_loader),
                         desc=f"Epoch {epoch+1}/{epochs} [val]", unit="pt", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["mask"].to(device, non_blocking=True).long()

                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = sliding_window_inference(
                        inputs=images, roi_size=(128, 128, 128),
                        sw_batch_size=4, predictor=model, overlap=0.25,
                    )
                    v_loss = criterion(logits, labels)

                val_loss_sum += v_loss.item()
                region = compute_brats_region_dice(logits, labels)
                dice_wt_sum += region["dice_wt"]
                dice_tc_sum += region["dice_tc"]
                dice_et_sum += region["dice_et"]
                per_class_sum += compute_dice_per_class(logits, labels).cpu()
                n_val += 1

                val_pbar.set_postfix(
                    dice_wt=f"{dice_wt_sum/n_val:.3f}",
                    dice_tc=f"{dice_tc_sum/n_val:.3f}",
                    dice_et=f"{dice_et_sum/n_val:.3f}",
                )

                del images, labels, logits, v_loss
        val_pbar.close()

        avg_val_loss = val_loss_sum / max(n_val, 1)
        dice_wt = dice_wt_sum / max(n_val, 1)
        dice_tc = dice_tc_sum / max(n_val, 1)
        dice_et = dice_et_sum / max(n_val, 1)
        mean_dice = (dice_wt + dice_tc + dice_et) / 3.0
        per_class = (per_class_sum / max(n_val, 1)).tolist()

        print(
            f"\n  \u2713 Epoch {epoch+1} | train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}\n"
            f"    Dice  WT={dice_wt:.4f}  TC={dice_tc:.4f}  ET={dice_et:.4f}  mean={mean_dice:.4f}"
        )

        entry = {
            "epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "dice_wt": dice_wt, "dice_tc": dice_tc, "dice_et": dice_et, "mean_dice": mean_dice,
            "dice_bg": per_class[0], "dice_ncr": per_class[1], "dice_ed": per_class[2], "dice_et_raw": per_class[3],
        }
        history.append(entry)

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(
                {"epoch": epoch + 1, "model": model.state_dict(), "mean_dice": mean_dice},
                str(output_dir / "best_finetune.pth"),
            )
            print(f"    \u2605 New best model saved  (mean_dice={mean_dice:.4f})")

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_mean_dice": best_mean_dice,
            },
            str(output_dir / "latest_finetune.pth"),
        )
        with open(output_dir / "finetune_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nFine-tuning complete. Best mean Dice: {best_mean_dice:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("pretrain")
    p1.add_argument("--data_root", required=False)
    p1.add_argument("--output_dir", default="./outputs")
    p1.add_argument("--epochs", type=int, default=100)
    p1.add_argument("--batch_size", type=int, default=2)
    p1.add_argument("--patch_size", type=int, default=16)
    p1.add_argument("--embed_dim", type=int, default=384)
    p1.add_argument("--encoder_depth", type=int, default=8)
    p1.add_argument("--encoder_heads", type=int, default=8)
    p1.add_argument("--lr", type=float, default=1.5e-4)
    p1.add_argument("--no_cross_modal", action="store_true")
    p1.add_argument("--resume", default=None)
    p1.add_argument("--num_workers", type=int, default=4)
    p1.add_argument("--cache_mode", choices=["precomputed", "persistent", "inmemory"],
                     default="precomputed",
                     help="precomputed (default, best for limited RAM): one explicit .pt "
                          "file per patient, built once. persistent: MONAI PersistentDataset "
                          "(implicit per-item disk cache). inmemory: MONAI CacheDataset "
                          "(cache_rate fraction held in RAM).")
    p1.add_argument("--force_recache", action="store_true",
                     help="Rebuild the precomputed cache even if files already exist "
                          "(use after changing the deterministic transform pipeline).")
    p1.add_argument("--cache_rate", type=float, default=0.0,
                     help="Only used when --cache_mode inmemory: fraction of the dataset "
                          "to hold in RAM via CacheDataset.")
    p1.add_argument("--cache_dir", default="monai_persistent_cache/pretrain")

    p2 = sub.add_parser("finetune")
    p2.add_argument("--data_root", required=False)
    p2.add_argument("--pretrained_encoder", required=True)
    p2.add_argument("--output_dir", default="./outputs")
    p2.add_argument("--epochs", type=int, default=100)
    p2.add_argument("--batch_size", type=int, default=2)
    p2.add_argument("--patch_size", type=int, default=16)
    p2.add_argument("--embed_dim", type=int, default=384)
    p2.add_argument("--encoder_depth", type=int, default=8)
    p2.add_argument("--encoder_heads", type=int, default=8)
    p2.add_argument("--lr", type=float, default=5e-5)
    p2.add_argument("--freeze_first_n", type=int, default=6)
    p2.add_argument("--num_workers", type=int, default=4)
    p2.add_argument("--cache_mode", choices=["precomputed", "persistent", "inmemory"],
                     default="precomputed",
                     help="precomputed (default, best for limited RAM): one explicit .pt "
                          "file per patient, built once. persistent: MONAI PersistentDataset "
                          "(implicit per-item disk cache). inmemory: MONAI CacheDataset "
                          "(cache_rate fraction held in RAM).")
    p2.add_argument("--force_recache", action="store_true",
                     help="Rebuild the precomputed cache even if files already exist "
                          "(use after changing the deterministic transform pipeline).")
    p2.add_argument("--cache_rate", type=float, default=0.0,
                     help="Only used when --cache_mode inmemory: fraction of the dataset "
                          "to hold in RAM via CacheDataset.")
    p2.add_argument("--train_cache_dir", default="monai_persistent_cache/finetune_train")
    p2.add_argument("--val_cache_dir", default="monai_persistent_cache/finetune_val")
    p2.add_argument("--num_samples_per_crop", type=int, default=1,
                     help="RandCropByPosNegLabeld num_samples; each sample multiplies "
                          "the effective batch memory footprint, so keep this low if RAM/VRAM-limited.")
    p2.add_argument("--resume", default=None,
                     help="Path to a latest_finetune.pth checkpoint to resume training from "
                          "(restores model, optimizer, scheduler, scaler, epoch, "
                          "best_mean_dice, and finetune_history.json).")

    args = parser.parse_args()

    if args.cmd == "pretrain":
        pretrain(
            data_root=args.data_root, output_dir=args.output_dir, epochs=args.epochs,
            batch_size=args.batch_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
            encoder_depth=args.encoder_depth, encoder_heads=args.encoder_heads, lr=args.lr,
            use_cross_modal=not args.no_cross_modal, resume=args.resume,
            num_workers=args.num_workers, cache_rate=args.cache_rate, cache_dir=args.cache_dir,
            cache_mode=args.cache_mode, force_recache=args.force_recache,
        )
    elif args.cmd == "finetune":
        finetune(
            data_root=args.data_root, pretrained_encoder=args.pretrained_encoder,
            output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
            patch_size=args.patch_size, embed_dim=args.embed_dim, encoder_depth=args.encoder_depth,
            encoder_heads=args.encoder_heads, lr=args.lr, freeze_first_n=args.freeze_first_n,
            num_workers=args.num_workers, cache_rate=args.cache_rate,
            train_cache_dir=args.train_cache_dir, val_cache_dir=args.val_cache_dir,
            num_samples_per_crop=args.num_samples_per_crop,
            cache_mode=args.cache_mode, force_recache=args.force_recache,
            resume=args.resume,
        )
    else:
        parser.print_help()
