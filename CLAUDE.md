# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CI2026 Hackathon Starter Kit for predicting **total cloud cover** on a global 64×64 grid at 1.5° resolution. Inputs are ERA5/AIMIP pressure-level fields (`input_level`: temperature, specific humidity, u/v wind at 7 levels) and static auxiliary fields (`input_auxiliary`: land-sea mask, orography, land cover, lon/lat). Target is daily-averaged total cloud cover ∈ [0, 1].

Scoring: MAE for ERA5 configurations, CRPS for AIMIP configurations (3 ensemble members required). Composite skill score = average of 4 region/dataset scores, normalised so Sundqvist baseline ≈ 0.

## Environment setup

```bash
conda env create -f environment.yml
conda activate ci26_starter_kit
uv pip install -r requirements.txt
pip install -e .
# Download data
hf download tobifinn/CI2026Hackathon --repo-type dataset --local-dir data/train_data
find data/train_data -name "*.zip" -exec unzip -o {} -d data/train_data \; -exec rm {} \;
```

## Key commands

```bash
# Train (default: baseline_mlp on CPU)
python scripts/train.py
python scripts/train.py device=cuda
python scripts/train.py +experiment=baseline_parametric device=cuda
python scripts/train.py +experiment=baseline_sundquist device=cuda

# Forecast for all 4 eval configurations (needed before evaluate/submit)
python scripts/forecast.py +suite=val +experiment=baseline_sundquist

# Evaluate locally (validation set only — no test targets available)
python scripts/evaluate.py --prediction_dir data/forecasts/baseline_sundquist
python scripts/evaluate.py --prediction_dir data/forecasts/baseline_sundquist --to_json --output_path scores/baseline_sundquist.json

# Submit to leaderboard (3 submissions/email/hour limit)
python scripts/submit.py --email <email> --experiments=baseline_sundquist
```

All scripts use Hydra: override any config key as `key=value` on the CLI. Use `python scripts/train.py --help` to see all options.

## Architecture

### Data flow

`data/train_data/` contains zarr archives (one per region/split). `TestDataset` / `TrainDataset` in `starter_kit/data.py` open them via **tensorstore** (lazy, per-worker forking). `input_auxiliary` is loaded into memory once; `input_level` (and `target` for training) are read on demand. Both datasets return dicts with keys `input_level`, `input_auxiliary`, and (for training) `target`.

`input_level` shape: `(C_level, n_levels, H, W)` — 4 vars × 7 levels = 28 channels.  
`input_auxiliary` shape: `(C_aux, H, W)` — 5 channels.  
`target` shape: `(1, H, W)`.

### Model abstraction

`BaseModel` (`starter_kit/model.py`) is an abstract trainer wrapping any `torch.nn.Module`. It owns the optimizer (AdamW), training/validation loops, checkpointing, and CSV logging. Subclasses must implement:
- `estimate_loss(batch) -> {"loss": ..., ...}` — called every step for both train and val
- `estimate_auxiliary_loss(batch, outputs) -> {...}` — optional, called only during validation; receives the `estimate_loss` output to avoid recomputation

`BaseModel.__call__` runs inference with `torch.inference_mode()` and clamps output to [0, 1].

Checkpoints are saved to `data/models/<exp_name>/best_model.ckpt` and `train_log.csv`.

### Config / Hydra wiring

`configs/train.yaml` composes `data/default`, `model/mlp`, `hydra/default`. The `model` key maps to a `_target_` that Hydra instantiates (e.g. `starter_kit.baselines.mlp.MLPModel`); `network` similarly. `+experiment=<name>` overlays `configs/experiments/<name>.yaml`, which typically sets both `model` and `network` targets.

### Implementing your own model

1. Write a `torch.nn.Module` subclass and a `BaseModel` subclass with `estimate_loss`.
2. Add `configs/model/my_model.yaml` with `_target_` pointing to your classes.
3. Optionally add `configs/experiments/my_experiment.yaml`.
4. Train: `python scripts/train.py model=my_model device=cuda`.

Use `InputNormalisation` (`starter_kit/layers.py`) for pre-computed mean/std stored as non-trainable buffers — they serialize with the checkpoint automatically.

Loss should be weighted by `self.lat_weights` (shape `(64, 1)`) from `BaseModel`, which corrects for unequal grid-cell areas at different latitudes.

### Forecast → evaluate → submit pipeline

`scripts/forecast.py +suite=val` reads `configs/suite/val.yaml` to loop over all 4 region/dataset combinations and writes `data/forecasts/<exp_name>/val_{era5,aimip}_region{1,2}.nc`. `evaluate.py` compares those to `data/train_data/val_target_*.nc`. `submit.py` runs the test suite (`+suite=test`) and POSTs to the leaderboard API.
