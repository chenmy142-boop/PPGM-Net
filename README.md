# PPGM-Net

PPGM-Net is a research implementation for the MUDCAD-X camouflaged object segmentation benchmark. This package includes the RGB segmentation model, offline prior generation and refinement, the experiment splits used in the paper, and evaluation scripts.

## Environment

The tested environment used Python 3.8, PyTorch 2.4.1 and torchvision 0.19.1. Install the PyTorch and torchvision builds appropriate for your system first. For the tested CUDA 11.8 setup:

```sh
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Training uses CUDA. For inference on CPU, append `--cpu` to the prediction command.

## Data

Obtain [MUDCAD-X](https://github.com/Tobias-UniBwM/MUDCAD-X) from the dataset authors. Place the season directories under `data/dataset/`. Keep their nested scene structure and match the filenames in `data/train.txt`, `data/val.txt` and `data/test.txt`. Evaluation uses `vis.png` and `label.png`; regenerating priors also requires `blue.png`, `green.png`, `red.png`, `nir.png`, `eir.png` and `lwir.png` alongside each visible image. The three splits contain 530, 107 and 216 images, respectively.

Run commands from the repository root. The demo launcher also accepts arguments forwarded to the selected script.

## Verify the PPGM-Net row of Table 3

The original training seed was not recorded. Use the archived checkpoint, its exact 216 refined test priors, and the archived predictions in the Table 3 artifacts. If downloaded separately, extract them to this layout:

```text
artifacts/table3/Net_epoch_best.pth
artifacts/table3/refined_priors/<sample>.png
artifacts/table3/predictions/<sample>.png
```

Checkpoint SHA-256: `1b59a21fe20064b7e4146f485ea274679cae381497d2acc1d417d2238ab119fd`.

Evaluate the archived maps:

```sh
python run_demo.py --mode eval
```

Regenerate and evaluate predictions:

```sh
python run_demo.py --mode predict --prior_dir artifacts/table3/refined_priors --save_dir results/table3_predictions
python run_demo.py --mode eval --ours_dir results/table3_predictions
```

Expected values at the reported precision:

| S-measure | max-F | adaptive E-measure | MAE (×10⁻³) | Target-level recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.844 | 0.848 | 0.948 | 1.12 | 0.911 |

Target-level recall uses grayscale threshold 128 and a target coverage threshold of 0.10, yielding 729 of 800 targets in the archived predictions. We verified byte-identical PNGs for all 216 predictions in the local tested GPU environment. Small floating-point differences may occur on other hardware.

## Run a new training experiment

```sh
python run_demo.py --mode prior --seed 42
python run_demo.py --mode clean
python run_demo.py --mode train --epoch 150 --seed 2024
python run_demo.py --mode predict --weights results/checkpoints/seed_2024/Net_epoch_best.pth --save_dir results/seed_2024
python run_demo.py --mode eval --ours_dir results/seed_2024
```

These commands define a new experiment. They do not recover the original training seed or reproduce the original RF sample draw. The two seeds above control different stages. Training selects the best checkpoint using validation MAE. If the Res2Net download is unavailable, give the official local pretrained file with `--backbone_weights`. Inference with a complete PPGM-Net checkpoint does not require a separate Res2Net file.

## Attribution

The encoder is based on [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels); the decoder follows the cascaded design of [SINet-v2](https://github.com/GewelsJI/SINet-V2). The paper describes the prior generation, refinement and modulation components.
