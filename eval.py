import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from monai.data import decollate_batch, DataLoader
from monai.transforms import Activations, AsDiscrete, Compose as MCompose
import monai

from core.data import build_transforms
from core.model import build_model, build_dice_metric
from core.uncertainty import TTAUncertaintyAnalyser
from utils.viz import save_visualization

log = logging.getLogger(__name__)

CLASS_NAMES = ["WT", "TC", "ET"]

def evaluate(checkpoint: str, output_dir: str = "seg3d_eval", mc_passes: int = 15,
            save_maps: bool = True, save_viz: bool = True):
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    viz_dir = out_path / "visualizations"
    if save_viz:
        viz_dir.mkdir(exist_ok=True)

    ckpt_path = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_size = tuple(cfg["roi_size"])

    val_files_path = ckpt_path / "val_goat.json"
    if not val_files_path.exists():
        raise FileNotFoundError(f"val_files.json not found in {ckpt_path.parent}. Run training first.")
    val_files = json.loads(val_files_path.read_text())
    log.info("Loaded %d val subjects", len(val_files))

    model = build_model(cfg["in_channels"], cfg["num_classes"],
                        dropout_p=cfg.get("dropout_p", 0.1)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Loaded checkpoint (epoch %d, best_dice=%.4f)",
             ckpt["epoch"], ckpt.get("best_mean_dice", 0.0))

    val_ds = monai.data.Dataset(data=val_files, transform=build_transforms(roi_size, train=False))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    dice_metric = build_dice_metric()
    post_pred = MCompose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    analyser = TTAUncertaintyAnalyser(model, roi_size=roi_size)

    all_results = []
    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float()
        pid = batch.get("label_meta_dict", {}).get("filename_or_obj", [f"subject_{i:03d}"])
        pid = Path(str(pid[0]) if isinstance(pid, (list, tuple)) else str(pid)).name.split(".")[0]

        unc = analyser.analyse(images)
        logits = torch.logit(unc["mean_pred"].clamp(1e-6, 1 - 1e-6))

        pred_disc = post_pred(decollate_batch(logits)[0])
        target = decollate_batch(labels)[0]
        dice_metric(y_pred=[pred_disc], y=[target])
        d_vals = dice_metric.get_buffer()[-1]  # per-channel dice for this subject
        d = {f"dice_{n}": d_vals[j].item() for j, n in enumerate(CLASS_NAMES)}
        d["mean_dice"] = float(np.mean(list(d.values())))

        result = {"subject_id": pid, **d,
                 "mean_variance": unc["variance"].mean().item(),
                 "mean_entropy": unc["entropy"].mean().item(),
                 "mean_mutual_info": unc["mutual_info"].mean().item()}
        all_results.append(result)
        log.info("[%s] Dice=%.4f  entropy=%.5f  MI=%.5f",
                 pid, d["mean_dice"], unc["entropy"].mean().item(), unc["mutual_info"].mean().item())

        if save_maps:
            subj_dir = out_path / pid
            subj_dir.mkdir(exist_ok=True)
            for key in ["mean_pred", "variance", "entropy", "mutual_info"]:
                np.save(str(subj_dir / f"{key}.npy"), unc[key][0].cpu().half().numpy())

        if save_viz:
            save_visualization(
                image=images[0], gt_mask=labels[0], pred_mask=pred_disc,
                uncertainty=unc["mutual_info"][0],
                out_file=str(viz_dir / f"{pid}.png"), pid=pid)

    dice_per_class = dice_metric.aggregate()
    overall = {f"dice_{n}": dice_per_class[j].item() for j, n in enumerate(CLASS_NAMES)}
    overall["mean_dice"] = float(np.mean(list(overall.values())))

    summary = {
        **{f"mean_{k}": float(np.mean([r[k] for r in all_results]))
           for k in all_results[0] if k != "subject_id"},
        "overall_dice": overall,
        "per_subject": all_results}
    (out_path / "eval_results.json").write_text(json.dumps(summary, indent=2))

    dice_str = "  ".join(f"{n}={overall[f'dice_{n}']:.4f}" for n in CLASS_NAMES)
    log.info("\n=== Evaluation Summary ===\n  Dice %s  mean=%.4f\n"
             "  mean_entropy=%.6f  mean_MI=%.6f\n  Saved to %s\n  Visualizations: %s",
             dice_str, overall["mean_dice"], summary["mean_mean_entropy"],
             summary["mean_mean_mutual_info"], out_path, viz_dir)
