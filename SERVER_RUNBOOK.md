# CLIPSeg PlantSeg Server Runbook

## Scope

This delivery targets a Linux server with:

- one RTX 4090 GPU
- CUDA 11.8
- Python 3.10
- Miniconda

The supported dataset is the sibling directory `../plantseg`.

## Dataset contract

- Keep the dataset next to the repository root as `../plantseg`.
- The dataset must contain:
  - `main.json`
  - `images/`
  - `ann/`
- The training and evaluation path uses `caption[3]` as the only text prompt.

## Environment setup

Create the conda environment from `environment.server.yml`.

- The environment is the source of truth for Linux server execution.
- `CLIPDensePredT(version='ViT-B/16')` downloads the official CLIP backbone automatically on first use.
- Do not place checkpoints, logs, or cached models inside Git-tracked paths.

## Training

Use the existing training entry with the PlantSeg experiment file:

```bash
python training.py plantseg.yaml 0
```

The delivered PlantSeg config is aligned to a 50-epoch comparison setup.

- with `9118` training samples and `batch_size=16`
- one epoch is `570` iterations
- `50` epochs map to `28500` iterations

Expected output:

- `logs/plantseg-rd64-vit16/config.json`
- `logs/plantseg-rd64-vit16/weights.pth`

If the first sanity run hits GPU OOM on the 4090, reduce the PlantSeg training batch size from `16` to `8` and recompute `max_iterations` / `T_max` so the run still covers 50 epochs.

## Post-training validation

Use the existing scoring entry with the PlantSeg experiment file:

```bash
python score.py plantseg.yaml 0
```

Behavior:

- loads `logs/plantseg-rd64-vit16/weights.pth`
- selects one threshold on the validation split by maximizing `mIoU`
- reports final test metrics with the chosen threshold

Reported metrics:

- `IoU`
- `Dice`
- `Recall`
- `mIoU`
- `mACC`
- `selected_threshold`
- `val_best_miou`

## Notes

- This server path does not require `third_party/` datasets or the repository's historical pretrained segmentation weights.
- The repository is adapted for Linux server execution first; local Windows execution is not the delivery target.
