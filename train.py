import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import Activations, AsDiscrete, Compose as MCompose

from core.data import build_dataloaders
from core.model import build_model, build_loss, build_dice_metric
from core.uncertainty import MCUncertaintyAnalyser

log = logging.getLogger(__name__)

CLASS_NAMES = ["WT", "TC", "ET"]

def train(
    dataset_json: str,
    output_dir:   str = "seg3d_out",
    roi_size:     Tuple[int, int, int] = (128, 128, 128),
    in_channels:  int = 4,
    num_classes:  int = 3,
    dropout_p:    float = 0.1,
    epochs:       int = 100,
    batch_size:   int = 2,
    lr:           float = 1e-4,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.2,
    num_workers:  int = 3,
    val_every:    int = 1,
    unc_every:    int = 5,
    mc_passes:    int = 15,
    seed:         int = 42,
    class_weights: Tuple[float, float, float] = (1.0, 1.0, 2.0),
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, val_files = build_dataloaders(
        dataset_json, roi_size, batch_size, num_workers, val_fraction, seed)
    (out_path / "val_files.json").write_text(json.dumps(val_files, indent=2, default=str))

    model = build_model(in_channels, num_classes, dropout_p=dropout_p).to(device)
    log.info("Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = build_loss(class_weights)
    criterion = criterion.to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dice_metric = build_dice_metric()
    analyser = MCUncertaintyAnalyser(model, n_passes=mc_passes, roi_size=roi_size)
    post_pred = MCompose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    cfg = {"dataset_json": dataset_json, "roi_size": list(roi_size),
          "in_channels": in_channels, "num_classes": num_classes,
          "dropout_p": dropout_p, "class_weights": list(class_weights)}
    (out_path / "config.json").write_text(json.dumps(cfg, indent=2))

    best_mean_dice = 0.0
    history: List[Dict] = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"E{epoch+1:>3}/{epochs} [train]",
                   dynamic_ncols=True, leave=False)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_loss /= max(len(train_loader), 1)
        entry = {"epoch": epoch + 1, "train_loss": train_loss}

        if (epoch + 1) % val_every == 0:
            model.eval()
            dice_metric.reset()
            run_unc = (epoch + 1) % unc_every == 0
            unc_sums = {"mean_variance": 0.0, "mean_entropy": 0.0}
            n_val = 0
            with torch.no_grad():
                for vb in tqdm(val_loader, desc=f"E{epoch+1:>3}/{epochs} [val]",
                              leave=False, dynamic_ncols=True):
                    vi = vb["image"].to(device)
                    vl = vb["label"].to(device).float()
                    if run_unc:
                        unc = analyser.analyse(vi)
                        logits = torch.logit(unc["mean_pred"].clamp(1e-6, 1 - 1e-6))
                        unc_sums["mean_variance"] += unc["variance"].mean().item()
                        unc_sums["mean_entropy"] += unc["entropy"].mean().item()
                    else:
                        logits = sliding_window_inference(vi, roi_size, 4, model, overlap=0.5)
                    preds = [post_pred(p) for p in decollate_batch(logits)]
                    targets = decollate_batch(vl)
                    dice_metric(y_pred=preds, y=targets)
                    n_val += 1

            dice_per_class = dice_metric.aggregate()
            dice_avg = {f"dice_{n}": dice_per_class[i].item()
                       for i, n in enumerate(CLASS_NAMES)}
            dice_avg["mean_dice"] = float(np.mean(list(dice_avg.values())))

            dice_str = "  ".join(f"{n}={dice_avg[f'dice_{n}']:.4f}" for n in CLASS_NAMES)
            unc_str = ("(skipped)" if not run_unc else
                      f"var={unc_sums['mean_variance']/max(n_val,1):.5f}  "
                      f"entropy={unc_sums['mean_entropy']/max(n_val,1):.5f}")
            log.info("Epoch %d/%d  train_loss=%.4f  [Dice] %s  mean=%.4f  [Unc] %s",
                     epoch + 1, epochs, train_loss, dice_str, dice_avg["mean_dice"], unc_str)

            entry.update({f"val_{k}": v for k, v in dice_avg.items()})
            if run_unc:
                entry.update({f"val_{k}": v / max(n_val, 1) for k, v in unc_sums.items()})

            if dice_avg["mean_dice"] > best_mean_dice:
                best_mean_dice = dice_avg["mean_dice"]
                torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                           "best_mean_dice": best_mean_dice, "config": cfg,
                           **dice_avg}, out_path / "best_model.pth")
                log.info("  * New best mean_dice=%.4f", best_mean_dice)
        else:
            log.info("Epoch %d/%d  train_loss=%.4f", epoch + 1, epochs, train_loss)

        history.append(entry)
        (out_path / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "config": cfg,
                   "best_mean_dice": best_mean_dice}, out_path / "latest.pth")

    log.info("Training complete. Best mean Dice: %.4f", best_mean_dice)
