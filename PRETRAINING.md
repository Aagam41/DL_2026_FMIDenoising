# Pretraining Workflow

End-to-end guide for: pretrain on RTX 5090 → load + fine-tune on T4 at submission.

## Why pretrain

Zero-shot training (training on each test stack from scratch) takes 60+ minutes per stack on T4 for the heavy models in this framework. With 7 test stacks and a 1-hour budget, that's infeasible. Pretraining once on training data and fine-tuning briefly per test stack drops per-stack cost to ~6-7 minutes.

The validation stack(s) with clean GT are used **only for model selection** — choosing which checkpoint to save as "best". No gradient ever flows from val.

## Supported algorithms

Two algos currently support pretrain → finetune via the `init_state_dict` parameter:

| Algo | Pretrain script | Pretrain config | Finetune config |
|---|---|---|---|
| `dvt_unet3d` | `scripts/pretrain_dvt.py` | `configs/dvt_unet3d_pretrain.py` | `configs/dvt_unet3d_finetune.py` |
| `restormer3d` | `scripts/pretrain_restormer3d.py` | `configs/restormer3d_pretrain.py` | `configs/restormer3d_finetune.py` |

For any other algo, `inference.py` falls back to load-only mode (no fine-tune) when a pretrained checkpoint is found. To add fine-tune support for another algo, add an `init_state_dict` parameter to that algo's `train_self_supervised` and copy/edit one of the pretrain scripts.

## Pipeline overview

```
┌───────────────────────────┐          ┌────────────────────────┐
│ TRAINING (offline, 5090)  │   →      │ INFERENCE (T4, submit) │
│                           │          │                        │
│ pretrain_<algo>.py reads: │          │ submission/ container  │
│   train_dir/   (noisy)    │   ckpt   │   loads pretrained,    │
│   val_noisy/  + clean/    │  ──────► │   fine-tunes 800 iters │
│ Writes:                   │          │   per stack, denoises  │
│   <algo>_best.pth ────────┘          └────────────────────────┘
│   pretraining_log.csv     │
└───────────────────────────┘
```

## Step 1 — pretraining (on RTX 5090)

### For DVT-UNet3D

```bash
python scripts/pretrain_dvt.py \
    --train-dir   /path/to/train_noisy_stacks \
    --val-noisy-dir /path/to/val/noisy \
    --val-clean-dir /path/to/val/clean \
    --output-dir  ./pretrained_dvt \
    --config      configs/dvt_unet3d_pretrain.py \
    --total-iters 50000 \
    --checkpoint-interval 500
```

### For Restormer3D

```bash
python scripts/pretrain_restormer3d.py \
    --train-dir   /path/to/train_noisy_stacks \
    --val-noisy-dir /path/to/val/noisy \
    --val-clean-dir /path/to/val/clean \
    --output-dir  ./pretrained_restormer3d \
    --config      configs/restormer3d_pretrain.py \
    --total-iters 50000 \
    --checkpoint-interval 500
```

### What happens (same for both)

- Loads every `*.tif` from `--train-dir` as a noisy training stack
- Loads paired noisy+clean stacks from `--val-noisy-dir` / `--val-clean-dir` (matched by filename stem)
- Builds the model matching the architecture in the config
- Trains using the existing 2-stage N2V flow, but with **multi-stack sampling** (each iter picks a random training stack)
- Every 500 iters: forwards val through `denoise_stack()`, computes stSNR vs clean
- If val stSNR improved, saves `<algo>_best.pth`
- Always saves `<algo>_last.pth`
- Appends one row to `pretraining_log.csv` per checkpoint

**Monitor** with:

```bash
tail -f ./pretrained_<algo>/pretraining_log.csv
```

The CSV columns are: `iter, phase, running_train_loss, val_stSNR_mean, val_stSNR_per_stack, val_stPSNR_mean, val_stSI_PSNR_mean, is_best, elapsed_sec`.

**File pairing**: val noisy and val clean files must have the same filename stem. E.g. `val/noisy/F0.tif` pairs with `val/clean/F0.tif`. Unmatched files are skipped with a warning.

**Architecture pinning**: pretrain and finetune configs MUST have identical architecture knobs. If they differ, the fine-tune load will fail loud with a clear error — silent partial-weight loading is a worse failure mode. The keys that must match:
- **DVT**: `base_ch`, `token_dim`, `grid_shape`, `n_vit_blocks`, `n_heads`, `normalization`, `temporal_target`
- **Restormer3D**: `dim`, `num_blocks`, `num_refinement_blocks`, `heads`, `ffn_expansion_factor`, `bias_free`, `normalization`, `temporal_target`

**Overwrite guard**: refuses to overwrite an existing `<algo>_best.pth` unless `--force-overwrite`.

### CLI options (both scripts)

| Flag | Default | What it does |
|---|---|---|
| `--total-iters` | 50000 | warmup + N2V iters combined |
| `--warmup-iters` | total // 25 | override warmup specifically |
| `--checkpoint-interval` | 500 | eval val + maybe save best every N iters |
| `--val-stacks-subset` | use all | only use first N val stacks (speeds up long runs) |
| `--seed` | 42 | reproducibility |
| `--force-overwrite` | off | bypass the best.pth overwrite guard |

## Step 2 — submission (on T4)

After pretraining finishes, copy the best checkpoint to the submission's model folder. **Filename matters** — `inference.py` looks for `<algo_name>.pth`:

```bash
# For DVT
cp pretrained_dvt/dvt_unet3d_best.pth submission/model/dvt_unet3d.pth

# For Restormer3D
cp pretrained_restormer3d/restormer3d_best.pth submission/model/restormer3d.pth
```

Build the submission image for the fine-tune config:

```bash
cd submission

# For DVT
./do_build.sh dvt_unet3d dvt_unet3d_finetune

# For Restormer3D
./do_build.sh restormer3d restormer3d_finetune

# Then:
./do_test_run.sh                       # local test before upload
./do_save.sh                           # creates uploadable .tar.gz
```

The `inference.py` will:
1. Detect the `.pth` in `/opt/ml/model/`
2. Load the state_dict
3. Call `train_self_supervised(..., init_state_dict=pretrained_state)` — which loads pretrained weights INTO the model, then runs the fine-tune iters with the lower fine-tune LR
4. Denoise the test stack

**Expected timing on T4** with the `_finetune` configs (`warmup_iters=0`, `n2v_iters=800`, `lr=1e-4`):
- Fine-tune:    ~5-6 min/stack
- Inference:    ~30-50 s/stack
- Total/stack:  ~6-7 min
- **7 stacks:   ~45-50 min ✓ under 1 hour**

## Three modes of `inference.py`

Set via `SUBMISSION_PRETRAINED_MODE` env var (default: `finetune`):

| `SUBMISSION_PRETRAINED_MODE` | Pretrained file present? | Behavior |
|---|---|---|
| `finetune` (default) | yes | Load weights → fine-tune `n2v_iters` from config → infer |
| `load` | yes | Load weights → skip training → infer (fastest, ~30s/stack) |
| any | no | Train from scratch → infer (slow, ~60+ min/stack) |

You'd usually use `finetune` for the submission. `load` (no fine-tune) is the fallback if you want maximum speed and trust the pretrained model fully — but it's risky on domain shift, since the model has zero adaptation to the test stack's specific noise statistics.

## Caveats (apply to both)

1. **Train/val/test distribution match** is the key assumption. Pretraining only generalizes to test stacks that come from the same imaging conditions. If the test set has very different SNR or noise statistics, fine-tuning helps but only so much.

2. **Val is used to choose iteration count.** Saving "best by val stSNR" is the cleanest possible use of val data, but it does mean iteration count is selected by val. With one or two val stacks, this signal is noisy. Use more val stacks if you have them.

3. **Per-stack normalization is recomputed** at inference time. Each test stack gets its own normalization params — the pretrained model is robust to this because it always saw normalized inputs.

4. **VRAM at pretrain time** with `batch_size=4`:
   - DVT (`base_ch=64`, patch 32×128×128): ~20-22 GB → fits 5090 (32GB)
   - Restormer3D (`dim=32`, patch 32×64×64): ~10-14 GB → fits 5090 comfortably
   - If you hit OOM, drop `batch_size` to 2.

5. **The first 500-1000 iters are throwaway.** Val stSNR will initially be very low (the model is essentially random). Don't panic — the log shows the curve climbing.

6. **NaN guards** in each algo's training loop are preserved. If you see "non-finite loss" warnings in the log, that's the guard catching an unstable batch. A few here and there is fine; if it's most batches, something's wrong (probably the data normalization).

7. **Restormer3D-specific**: the temporal-overlap fix (50% overlap + Hann window + mirror-padding the time axis) is preserved at inference. The denoise log shows `temporal overlap 50%` and `mirror-pad time +N` to confirm.

## Provenance

After pretraining, `pretrained_<algo>/pretraining_config.json` contains:
- Full CLI args
- Full config dict used
- List of training stacks
- List of val pairs
- Total model parameters
- Device + GPU name
- Start timestamp

You can use this to reproduce the run exactly.
