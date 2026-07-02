import sys
import logging
import argparse
from pathlib import Path

from train import train
from eval import evaluate

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Simplified 3D Segmentation Pipeline",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--dataset_json", required=True)
    t.add_argument("--output_dir", required=True)
    t.add_argument("--roi_size", type=int, nargs=3, default=[128, 128, 128])
    t.add_argument("--in_channels", type=int, default=4)
    t.add_argument("--num_classes", type=int, default=3)
    t.add_argument("--dropout_p", type=float, default=0.1)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch_size", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight_decay", type=float, default=1e-5)
    t.add_argument("--val_fraction", type=float, default=0.2)
    t.add_argument("--num_workers", type=int, default=0)
    t.add_argument("--val_every", type=int, default=1)
    t.add_argument("--unc_every", type=int, default=10,
                   help="Run MC-dropout uncertainty during validation every N epochs.")
    t.add_argument("--mc_passes", type=int, default=15,
                   help="Number of stochastic forward passes for MC-dropout uncertainty.")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--class_weights", type=float, nargs=3, default=[1.0, 1.0, 2.0],
                   help="Per-class loss weights in WT, TC, ET order.")
    t.add_argument("--log_level", default="INFO")

    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True,)
    e.add_argument("--output_dir", required=True)
    e.add_argument("--mc_passes", type=int, default=15)
    e.add_argument("--no_save_maps", action="store_true")
    e.add_argument("--no_save_viz", action="store_true",
                   help="Skip saving tumor/uncertainty visualization PNGs.")
    e.add_argument("--log_level", default="INFO")

    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.cmd is None:
        parse_args(["--help"]); return

    out_dir = getattr(args, "output_dir", "seg3d_out")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                 logging.FileHandler(Path(out_dir) / "seg3d.log", mode="a")])

    if args.cmd == "train":
        train(dataset_json=args.dataset_json, output_dir=args.output_dir,
             roi_size=tuple(args.roi_size), in_channels=args.in_channels,
             num_classes=args.num_classes, dropout_p=args.dropout_p,
             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             weight_decay=args.weight_decay, val_fraction=args.val_fraction,
             num_workers=args.num_workers, val_every=args.val_every,
             unc_every=args.unc_every, mc_passes=args.mc_passes,
             seed=args.seed, class_weights=tuple(args.class_weights))
    elif args.cmd == "eval":
        evaluate(checkpoint=args.checkpoint, output_dir=args.output_dir,
                mc_passes=args.mc_passes, save_maps=not args.no_save_maps,
                save_viz=not args.no_save_viz)

if __name__ == "__main__":
    main()
