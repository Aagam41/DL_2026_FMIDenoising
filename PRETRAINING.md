# DVT-UNet3D Pretraining Workflow

End-to-end guide for: pretrain on RTX 5090 → load + fine-tune on T4 at submission.

## Why pretrain

DVT's zero-shot N2V training on a single stack takes 60+ minutes on T4 for ~2000 iters. With 7 test stacks and a 1-hour budget, that's infeasible. Pretraining once on training data and fine-tuning briefly per test stack drops per-stack cost to ~6–7 minutes.

The validation stack(s) with clean GT are used **only for model selection** — choosing which checkpoint to save as "best". No gradient ever flows from val.

## Pipeline overview

```
┌──────────────────────────┐          ┌────────────────────────┐
│ TRAINING (offline, 5090) │   →      │ INFERENCE (T4, submit) │
│                          │          │                        │
│ pretrain_dvt.py reads:   │          │ submission/ container  │
│   train_dir/   (noisy)   │   ckpt   │   loads pretrained,    │
│   val_noisy/  + clean/   │  ──────► │   fine-tunes 800 iters │
│ Writes:                  │          │   per stack, denoises  │
│   dvt_unet3d_best.pth ───┘          └────────────────────────┘
│   pretraining_log.csv    │
└──────────────────────────┘
```

## Step 1 — pretraining (on RTX 5090)

```bash
python scripts/pretrain_dvt.py \
    --train-dir   /path/to/train_noisy_stacks \
    --val-noisy-dir /path/to/val/noisy \
    --val-clean-dir /path/to/val/clean \
    --output-dir  ./pretrained \
    --config      configs/dvt_unet3d_pretrain.py \
    --total-iters 50000 \
    --checkpoint-interval 500
```

**What happens**:
- Loads every `*.tif` from `--train-dir` as a noisy training stack
- Loads paired noisy+clean stacks from `--val-noisy-dir` / `--val-clean-dir` (matched by filename stem)
- Builds DVT-UNet3D matching `configs/dvt_unet3d_pretrain.py` architecture
- Trains using the existing 2-stage N2V flow, but with **multi-stack sampling** (each iter picks a random training stack)
- Every 500 iters: forwards val through `denoise_stack()`, computes stSNR vs clean
- If val stSNR improved, saves `pretrained/dvt_unet3d_best.pth`
- Always saves `pretrained/dvt_unet3d_last.pth`
- Appends one row to `pretrained/pretraining_log.csv` per checkpoint

**Monitor** with:

```bash
tail -f ./pretrained/pretraining_log.csv
```

The CSV columns are: `iter, phase, running_train_loss, val_stSNR_mean, val_stSNR_per_stack, val_stPSNR_mean, val_stSI_PSNR_mean, is_best, elapsed_sec`.

**File pairing**: val noisy and val clean files must have the same filename stem. E.g. `val/noisy/F0.tif` pairs with `val/clean/F0.tif`. Unmatched files are skipped with a warning.

**Architecture pinning**: `configs/dvt_unet3d_pretrain.py` and `configs/dvt_unet3d_finetune.py` MUST have identical architecture knobs (base_ch, token_dim, grid_shape, n_vit_blocks, n_heads, normalization, temporal_target). If they differ, the fine-tune load will fail loud with a clear error — this is intentional, silent partial-weight loading is a worse failure mode.

**Overwrite guard**: the script refuses to overwrite an existing `dvt_unet3d_best.pth` unless `--force-overwrite`. Easy way to avoid losing a good run by accident.

**CLI options**:

| Flag | Default | What it does |
|---|---|---|
| `--total-iters` | 50000 | warmup + N2V iters combined |
| `--warmup-iters` | total // 25 | override warmup specifically |
| `--checkpoint-interval` | 500 | eval val + maybe save best every N iters |
| `--val-stacks-subset` | use all | only use first N val stacks (speeds up long runs) |
| `--seed` | 42 | reproducibility |
| `--force-overwrite` | off | bypass the best.pth overwrite guard |

## Step 2 — submission (on T4)

After pretraining finishes, copy `pretrained/dvt_unet3d_best.pth` to `submission/model/dvt_unet3d.pth`:

```bash
cp pretrained/dvt_unet3d_best.pth submission/model/dvt_unet3d.pth
```

Build the submission image for the fine-tune config:

```bash
cd submission
./do_build.sh dvt_unet3d dvt_unet3d_finetune
./do_test_run.sh                       # local test before upload
./do_save.sh                           # creates uploadable .tar.gz
```

The `inference.py` will:
1. Detect `dvt_unet3d.pth` in `/opt/ml/model/`
2. Load the state_dict
3. Call `train_self_supervised(..., init_state_dict=pretrained_state)` — which loads the pretrained weights INTO the model, then runs 800 N2V iters with the lower fine-tune LR
4. Denoise the test stack

**Expected timing on T4** with `dvt_unet3d_finetune.py` (warmup=0, n2v=800, lr=1e-4, batch=2):
- Fine-tune:    ~5–6 min/stack
- Inference:    ~30–50 s/stack
- Total/stack:  ~6–7 min
- **7 stacks:   ~45–50 min ✓ under 1 hour**

## Three modes of `inference.py`

Set via `SUBMISSION_PRETRAINED_MODE` env var (default: `finetune`):

| `SUBMISSION_PRETRAINED_MODE` | Pretrained file present? | Behavior |
|---|---|---|
| `finetune` (default) | yes | Load weights → fine-tune `n2v_iters` from config → infer |
| `load` | yes | Load weights → skip training → infer (fastest, ~30s/stack) |
| any | no | Train from scratch → infer (slow, ~60+ min/stack) |

You'd usually use `finetune` for the submission. `load` (no fine-tune) is the fallback if you want maximum speed and trust the pretrained model fully — but it's risky on domain shift, since the model has zero adaptation to the test stack's specific noise statistics.

## Caveats

1. **Train/val/test distribution match** is the key assumption. Pretraining only generalizes to test stacks that come from the same imaging conditions. If the test set has very different SNR or noise statistics, fine-tuning helps but only so much.

2. **Val is used to choose iteration count.** Saving "best by val stSNR" is the cleanest possible use of val data, but it does mean iteration count is selected by val. With one or two val stacks, this signal is noisy. Use more val stacks if you have them.

3. **Per-stack normalization is recomputed** at inference time. Each test stack gets its own normalization params (p3-p97 percentiles of that stack) — the pretrained model is robust to this because it always saw normalized inputs.

4. **VRAM at pretrain time**: with `batch_size=4`, `patch=32×128×128`, `base_ch=64`, expected VRAM is ~20-22 GB. Fits on 5090 (32GB). If you hit OOM, drop `batch_size` to 2 or shrink `patch_d` to 24.

5. **The first 500-1000 iters are throwaway**. Val stSNR will initially be very low (the model is essentially random). Don't panic — the log shows the curve climbing.

6. **NaN guards** in DVT's training loop are preserved. If you see "non-finite loss" warnings in the log, that's the guard catching an unstable batch. A few here and there is fine; if it's most batches, something's wrong (probably the data normalization).

## Provenance

After pretraining, `pretrained/pretraining_config.json` contains:
- Full CLI args
- Full config dict used
- List of training stacks
- List of val pairs
- Total model parameters
- Device + GPU name
- Start timestamp

You can use this to reproduce the run exactly.
