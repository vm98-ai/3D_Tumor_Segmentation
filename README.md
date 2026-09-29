# 3D Medical Image Segmentation Pipeline (BraTS 2023)

A simplified, robust 3D medical image segmentation pipeline for the BraTS 2023 GLI (Adult Glioma) dataset, built with PyTorch and MONAI. The repo contains **two complementary tracks**:

| Track | Location | What it does |
|-------|----------|--------------|
| **Supervised baseline** | `main.py` | MONAI 3D UNet trained from scratch, with TTA and Monte Carlo dropout uncertainty |
| **4D-JEPA self-supervised** | `jepa/` | Self-supervised pretraining of a 3D Vision Transformer / Mamba encoder on unlabeled MRI, then fine-tuning with a UNETR-style decoder |

---

## 🚀 Features

### Supervised baseline
- **Framework:** PyTorch + MONAI (`monai.networks.nets.UNet`).
- **BraTS 2023 label mapping** (1=NCR/NET, 2=ED/SNFH, 3=ET) into target regions:
  - **WT** (Whole Tumor): labels 1, 2, 3
  - **TC** (Tumor Core): labels 1, 3
  - **ET** (Enhancing Tumor): label 3
- **Loss:** `DiceCELoss` with per-channel weighting (default WT=1, TC=1, ET=2) to prioritize the small ET class.
- **Uncertainty estimation:**
  - *Monte Carlo Dropout* during validation: mean prediction, voxelwise variance, predictive entropy.
  - *Test-Time Augmentation (TTA)*: 8 deterministic flip combinations across X, Y, Z for epistemic/aleatoric uncertainty.
- **Visualization:** 4-panel PNGs (base image, ground truth, prediction, uncertainty heatmap).

### 4D-JEPA (self-supervised) 🆕
- **JEPA-style pretraining** (Joint-Embedding Predictive Architecture): predicts the *latent representations* of masked 3D regions, not raw voxels. No labels needed.
- **4-modality input:** T1n, T1c, T2w, T2f stacked as a `[B, 4, H, W, D]` volume, tokenized into 3D patches.
- **Two pretraining objectives:**
  - **Spatial JEPA**: predict EMA-target-encoder embeddings of masked 3D blocks from the visible context.
  - **Cross-modal JEPA**: predict T2f (FLAIR) token embeddings from T1c token embeddings.
- **Swappable backbone:** standard multi-head **attention** blocks or **Mamba** (selective state-space) blocks via `--block_type`.
- **UNETR-style decoder** for fine-tuning: fuses features from several intermediate encoder depths plus a full-resolution input skip.
- **Feature diagnostic:** PCA-of-features visualization to check the frozen encoder actually captures anatomy.
- **Fast data loading:** per-patient precomputed `.pt` cache (float16) for low-RAM machines.

---

## 📦 Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- MONAI
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- nibabel *(required by the JEPA scripts)*

```bash
pip install torch monai numpy matplotlib scikit-learn tqdm nibabel
```

---

## 📁 Project Structure

```
.
├── main.py                     # Supervised UNet baseline (train / eval)
├── README.md
└── jepa/                       # 4D-JEPA self-supervised track
    ├── tokenizer.py            # 3D patch embedding, 3D sin-cos pos. embedding, multi-block masker
    ├── model.py                # ViT/Mamba encoder, EMA target encoder, predictors, UNETR decoder
    ├── train_jepa.py           # Pretraining + fine-tuning CLI
    └── visualize_jepa.py       # PCA-of-features sanity check
```

---

## 📂 Dataset Structure

### Supervised baseline (`main.py`)

Uses a Medical Segmentation Decathlon-style `dataset.json`:

```json
{
  "training": [
    {
      "t2f": "BraTS-GLI-00001-000/BraTS-GLI-00001-000-t2f.nii.gz",
      "t1n": "BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1n.nii.gz",
      "t1c": "BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1c.nii.gz",
      "t2w": "BraTS-GLI-00001-000/BraTS-GLI-00001-000-t2w.nii.gz",
      "label": "BraTS-GLI-00001-000/BraTS-GLI-00001-000-seg.nii.gz"
    }
  ]
}
```

### JEPA track (`jepa/train_jepa.py`)

Does **not** use `dataset.json`. Point `--data_root` at a folder of per-patient subfolders; file names are derived from the folder name:

```
data_root/
├── BraTS-GLI-00001-000/
│   ├── BraTS-GLI-00001-000-t1n.nii.gz
│   ├── BraTS-GLI-00001-000-t1c.nii.gz
│   ├── BraTS-GLI-00001-000-t2w.nii.gz
│   ├── BraTS-GLI-00001-000-t2f.nii.gz
│   └── BraTS-GLI-00001-000-seg.nii.gz     # only needed for fine-tuning
├── BraTS-GLI-00002-000/
└── ...
```

> **Channel order** is fixed to `(t1n, t1c, t2w, t2f)` = channels `0, 1, 2, 3`. The cross-modal objective relies on this (T1c = channel 1, T2f = channel 3). Patients with mismatched modality shapes are automatically detected and skipped.

---

## 🛠️ Usage: Supervised Baseline

### Training

```bash
python main.py train --dataset_json /path/to/dataset.json --output_dir ./seg3d_out
```

Key arguments:
- `--epochs`: number of training epochs (default: 100)
- `--batch_size`: batch size (default: 2)
- `--lr`: learning rate (default: 1e-4)
- `--roi_size`: spatial crop size for training (default: 128 128 128)
- `--unc_every`: run MC-dropout uncertainty during validation every N epochs (default: 10)
- `--class_weights`: per-class loss weights for WT, TC, ET (default: 1.0 1.0 2.0)

### Evaluation

```bash
python main.py eval --checkpoint ./seg3d_out/best_model.pth --output_dir ./seg3d_eval
```

Key arguments:
- `--mc_passes`: number of stochastic forward passes for evaluation (default: 15)
- `--no_save_maps`: disable saving raw numpy arrays of uncertainty maps
- `--no_save_viz`: skip generating `.png` visual overlays

---

## 🧠 4D-JEPA: Self-Supervised Pretraining + Fine-tuning

### How it works

**Phase 1: Pretraining (no labels)**

1. **Masking.** `MultiBlock3DMasker` samples several contiguous 3D blocks (default 6 blocks, each 15–35 % of each axis) until ~65 % of patches are masked. Masked patches are the *targets*; the rest are the *context*.
2. **Context encoder** processes only the visible tokens (`forward_subset`), which is cheaper than masking in pixel space.
3. **EMA target encoder** (an exponential-moving-average copy of the encoder, no gradients) embeds the full volume. τ is annealed from 0.996 → 1.0.
4. **Spatial predictor** (a narrow transformer) takes the context embeddings plus learnable mask tokens carrying the target positions, and predicts the target encoder's embeddings at the masked locations.
5. **Loss.** L2-normalized MSE between predicted and target embeddings (targets detached).
6. **Cross-modal objective (optional).** A shared single-channel patch projection encodes T1c; a cross-attention predictor predicts the EMA encoder's T2f embeddings. Total loss = `spatial + 0.25 × cross_modal`. Disable with `--no_cross_modal`.

**Phase 2: Fine-tuning (labels)**

The pretrained encoder is loaded and a **UNETR-style decoder** is attached. Hidden states from `log2(patch_size)` evenly spaced encoder blocks (e.g. 4 for `patch_size=16`, printed at startup as "UNETR decoder skip layers") are reshaped to 3D grids, upsampled, and fused with a full-resolution skip from the raw input. The first `--freeze_first_n` encoder blocks (and the patch embedding) are frozen by default. Loss is `0.5 × Dice + 0.5 × CrossEntropy`, and validation uses sliding-window inference (128³ ROI, 25 % overlap) reporting WT / TC / ET Dice.

### Encoder options

| `--block_type` | Mixer | Complexity | Notes |
|----------------|-------|------------|-------|
| `attention` (default) | Multi-head self-attention | O(N²) | Standard ViT |
| `mamba` | Bidirectional selective SSM (S6) | O(N) | Useful for small patch sizes (many tokens). Pure-PyTorch reference scan, so it's correct but slow. Swap in `mamba_ssm`'s fused kernel for production speed. |

> ⚠️ `--block_type` (and `--patch_size`, `--embed_dim`, `--encoder_depth`, `--encoder_heads`) **must match** between pretraining, fine-tuning and visualization, or the checkpoint will not load.

### Step 1: Pretrain

```bash
python jepa/train_jepa.py pretrain \
    --data_root /path/to/BraTS_data \
    --output_dir ./jepa_outputs \
    --epochs 100 \
    --batch_size 2 \
    --patch_size 16 --embed_dim 384 --encoder_depth 8 --encoder_heads 8 \
    --block_type attention
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | – | Folder of per-patient subfolders (see above) |
| `--output_dir` | `./outputs` | Where checkpoints and history are written |
| `--epochs` | 100 | Pretraining epochs (20 warm-up epochs, then cosine decay) |
| `--batch_size` | 2 | Batch size |
| `--lr` | 1.5e-4 | Peak learning rate (AdamW, weight decay 0.05) |
| `--patch_size` | 16 | Cube side in voxels; must be a power of 2 and divide 128 |
| `--embed_dim` / `--encoder_depth` / `--encoder_heads` | 384 / 8 / 8 | Encoder size |
| `--block_type` | `attention` | `attention` or `mamba` |
| `--mamba_d_state` / `--mamba_d_conv` / `--mamba_expand` | 16 / 4 / 2 | Mamba hyperparameters |
| `--no_cross_modal` | off | Disable the T1c→T2f objective |
| `--resume` | – | Path to `latest_pretrain.pth` to resume |
| `--num_workers` | 4 | DataLoader workers |
| `--cache_mode` | `precomputed` | See "Caching" below |
| `--cache_dir` | `monai_persistent_cache/pretrain` | Cache location |
| `--cache_rate` | 0.0 | Fraction held in RAM (only for `inmemory`) |
| `--force_recache` | off | Rebuild cache after changing the transform pipeline |

**Outputs** (in `--output_dir`):
- `pretrained_encoder.pth`: final encoder `state_dict` (**use this for fine-tuning**)
- `latest_pretrain.pth`: full checkpoint (model, optimizer, epoch) for `--resume`
- `pretrain_history.json`: per-epoch `loss`, `spatial_loss`, `cross_modal_loss`

Preprocessing: per-channel z-score over nonzero voxels → crop to foreground (8-voxel margin) → pad to 128³ → random 128³ crop. 1 % of patients are held out in pretraining.

### Step 2: Fine-tune for segmentation

```bash
python jepa/train_jepa.py finetune \
    --data_root /path/to/BraTS_data \
    --pretrained_encoder ./jepa_outputs/pretrained_encoder.pth \
    --output_dir ./jepa_finetune \
    --epochs 100 \
    --patch_size 16 --embed_dim 384 --encoder_depth 8 --encoder_heads 8 \
    --block_type attention \
    --freeze_first_n 6
```

Key arguments (in addition to the shared architecture/caching flags above):

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_encoder` | required | Path to `pretrained_encoder.pth` |
| `--lr` | 5e-5 | Learning rate (AdamW + cosine annealing) |
| `--freeze_first_n` | 6 | Freeze patch embedding + first N encoder blocks. Use `0` to fine-tune everything |
| `--num_samples_per_crop` | 1 | Crops per volume; each one multiplies memory use |
| `--train_cache_dir` / `--val_cache_dir` | `monai_persistent_cache/finetune_{train,val}` | Cache locations |
| `--resume` | – | Path to `latest_finetune.pth` (restores model, optimizer, scheduler, scaler, epoch, best Dice, history) |

**Outputs** (in `--output_dir`):
- `best_finetune.pth`: best checkpoint by mean (WT, TC, ET) Dice
- `latest_finetune.pth`: full resumable checkpoint
- `finetune_history.json`: per-epoch losses, WT/TC/ET Dice, and per-class Dice

Fine-tuning uses an 80/20 train/val split (`random_state=42`) and BraTS-style augmentation (Gaussian noise, contrast, bias field, affine, elastic, and positive/negative-biased cropping).

### Caching

Loading and preprocessing NIfTI volumes on the fly is a bottleneck. `--cache_mode` picks a strategy:

| Mode | Behavior | Best for |
|------|----------|----------|
| `precomputed` (default) | Builds one float16 `.pt` file per patient once, then loads them | Limited RAM |
| `persistent` | MONAI `PersistentDataset`: implicit per-item disk cache | Convenience |
| `inmemory` | MONAI `CacheDataset`: keeps `--cache_rate` of data in RAM | Plenty of RAM |

The precompute step prints an estimated disk-space requirement and warns if it won't fit. Re-run with `--force_recache` if you change the deterministic preprocessing.

### Step 3 (optional but recommended): Sanity-check the pretrained encoder

`visualize_jepa.py` runs the **frozen** encoder on a full volume, projects the per-patch token embeddings to their top 3 principal components, and paints them as an RGB image over the patch grid, side-by-side with the T1c slice. If pretraining worked, you should see anatomically coherent structure (brain tissue, ventricles, tumor regions) and not noise or a flat color.

```bash
# From a precomputed cache file (preferred)
python jepa/visualize_jepa.py \
    --encoder_ckpt ./jepa_outputs/pretrained_encoder.pth \
    --cache_pt monai_persistent_cache/pretrain/BraTS-GLI-00001-000.pt \
    --patch_size 16 --embed_dim 384 --encoder_depth 8 --encoder_heads 8 \
    --block_type attention \
    --output_dir feature_viz

# Or from raw NIfTI files
python jepa/visualize_jepa.py \
    --encoder_ckpt ./jepa_outputs/pretrained_encoder.pth \
    --t1n BraTS-XXX-t1n.nii.gz --t1c BraTS-XXX-t1c.nii.gz \
    --t2w BraTS-XXX-t2w.nii.gz --t2f BraTS-XXX-t2f.nii.gz \
    --patch_size 16 --embed_dim 384 --encoder_depth 8 --encoder_heads 8
```

**Outputs** (in `--output_dir`, default `feature_viz`):
- `pca_features.png`: grid of T1c slices vs. PCA-RGB slices (axial / coronal / sagittal, 5 levels each)
- `pca_grid.npy`: raw `[gh, gw, gd, 3]` PCA volume for your own analysis
- Console: explained-variance ratio of the top 3 components and the foreground patch fraction

The PCA is fit on **foreground patches only** (background is painted black), so the empty background doesn't dominate the color scale. Use `--fg_threshold` to adjust foreground detection and `--slices` to choose specific slice indices.

---

## 📄 License

This project is licensed under the MIT License.
