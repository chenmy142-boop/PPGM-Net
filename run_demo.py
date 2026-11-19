"""Run documented workflows from the repository root."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "prior", "clean", "train", "predict", "eval"), default="check")
    args, extra = parser.parse_known_args()
    if args.mode == "check":
        if extra:
            parser.error(f"Unknown arguments: {extra}")
        paths = ("data/train.txt", "data/val.txt", "data/test.txt", "data/dataset",
                 "artifacts/table3/Net_epoch_best.pth", "artifacts/table3/predictions",
                 "data/priors/cleaned")
        for name in paths:
            print(f"{'FOUND' if (ROOT / name).exists() else 'NOT INSTALLED'} {name}")
        return
    script, defaults = {
        "prior": ("generate_prior.py", ["--save_dir", "data/priors/raw", "--model_save_path", "data/priors/rf_prior_model.pkl"]),
        "clean": ("clean_physical.py", ["--input_dir", "data/priors/raw", "--output_dir", "data/priors/cleaned"]),
        "train": ("train.py", []),
        "predict": ("predict.py", []),
        "eval": ("evaluate_metrics.py", ["--ours_dir", "artifacts/table3/predictions"]),
    }[args.mode]
    subprocess.run([sys.executable, f"src/{script}", *defaults, *extra], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
