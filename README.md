# 3D Medical Image Segmentation Pipeline (BraTS 2023)

A simplified, robust 3D medical image segmentation pipeline designed for the **BraTS 2023 GLI (Adult Glioma)** dataset. Built using PyTorch and MONAI, this project features robust data augmentation, test-time augmentation (TTA), Monte Carlo dropout uncertainty estimation, and intuitive visualization tools.

## 🚀 Features
- **Framework**: Built natively on PyTorch and MONAI (using `monai.networks.nets.UNet`).
- **BraTS 2023 Standards**: Automatically maps labels (1=NCR/NET, 2=ED/SNFH, 3=ET) into logical target classes:
  - **WT (Whole Tumor)**: labels 1, 2, 3
  - **TC (Tumor Core)**: labels 1, 3
  - **ET (Enhancing Tumor)**: label 3
- **Robust Loss Strategy**: Utilizes `DiceCELoss` with per-channel weighting (default: WT=1, TC=1, ET=2) to prioritize the smaller/rare ET class.
- **Advanced Uncertainty Estimation**:
  - **Monte Carlo Dropout**: Performs stochastic forward passes during training validation to report mean prediction, voxelwise variance, and predictive entropy.
  - **Test-Time Augmentation (TTA)**: 8 deterministic spatial flip combinations across X, Y, and Z axes during evaluation to calculate robust epistemic and aleatoric uncertainty.
- **Automated Visualization**: Generates side-by-side 4-panel PNGs for evaluation containing:
  1. Base Image
  2. Ground Truth Overlay
  3. Prediction Overlay
  4. Uncertainty (Entropy/Mutual Info) Heatmap

## 📦 Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- MONAI
- NumPy
- Matplotlib
- scikit-learn
- tqdm

Install dependencies using:
```bash
pip install torch monai numpy matplotlib scikit-learn tqdm
```

## 📂 Dataset Structure
The script expects a dataset referenced via a Medical Segmentation Decathlon-style `dataset.json` file. 

Example structure inside `dataset.json`:
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

## 🛠️ Usage

### Training
Start the training process by pointing to your `dataset.json`. 
```bash
python main.py train --dataset_json /path/to/dataset.json --output_dir ./seg3d_out
```

**Key Arguments:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--roi_size`: Spatial crop size for training (default: 128 128 128)
- `--unc_every`: Run MC-dropout uncertainty during validation every N epochs (default: 10).
- `--class_weights`: Per-class loss weights for WT, TC, ET (default: 1.0 1.0 2.0).

### Evaluation
Evaluate the trained model and automatically generate multi-panel visualizations:
```bash
python main.py eval --checkpoint ./seg3d_out/best_model.pth --output_dir ./seg3d_eval
```

**Key Arguments:**
- `--mc_passes`: Number of stochastic forward passes for evaluation (default: 15).
- `--no_save_maps`: Flag to disable saving raw numpy arrays of uncertainty maps.
- `--no_save_viz`: Flag to skip generating and saving `.png` visual overlays.

## 📄 License
This project is licensed under the MIT License.
