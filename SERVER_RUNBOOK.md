# CLIPSeg Server Runbook

## Directory layout

On the Linux server, keep the repository and dataset as siblings:

```text
<workspace>/
  clipseg/
  dataset/
```

The repository reads the custom dataset from `../dataset` relative to the repository root.

## Dataset contract

- Training index: `../dataset/train.json`
- Test index: `../dataset/test.json`
- Each sample must provide `id`, `image`, `mask`, `caption`
- Training and test both use `caption[2]`
- Mask semantics are fixed to binary: `0` is background, any non-zero value is foreground

## Environment setup

Create the Linux GPU environment from `environment.server.gpu.yml`.

```bash
conda env create -f environment.server.gpu.yml
conda activate clipseg-server-gpu
```

## Official weights

Download the official CLIPSeg decoder weights into `weights/rd64-uni.pth`:

```bash
python tools/download_official_weights.py
```

If the server cannot access the internet, place the official `rd64-uni.pth` file at `weights/rd64-uni.pth` manually before training.

This download is only the official CLIPSeg decoder checkpoint. The OpenAI CLIP `ViT-B/16` backbone is not bundled in that file and is still fetched by `clip.load('ViT-B/16')` on first use. If the server is offline, provide the CLIP cache manually.

## Training

Train with the custom binary text dataset using the new experiment config:

```bash
python training.py sibling_binary_text.yaml 0
```

This uses:

- `datasets.sibling_binary_text.SiblingBinaryTextDataset`
- `models.clipseg.CLIPDensePredT`
- no validation split or validation checkpoint selection
- official `weights/rd64-uni.pth` as initialization weights
- larger default throughput settings for a single RTX 4090:
  - train `batch_size=32`
  - test `batch_size=64`
  - train `num_workers=12`
  - fixed text embedding cache enabled

Training outputs are written to `logs/custom-rd64-uni-bin/`.

## Post-training validation

Run the formal test split evaluation against the final checkpoint:

```bash
python score.py sibling_binary_text.yaml 0
```

The reported metrics are:

- `iou`
- `dice`
- `recall`
- `miou`
- `macc`

`miou` and `macc` are both computed as the mean over background and foreground classes.

## Notes

- This delivery is Linux-server-first and not designed around local Windows execution.
- No validation split is used.
- The final formal test run always uses the final `logs/<run>/weights.pth`.
