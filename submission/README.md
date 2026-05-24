# CIDC25 Submission Builder

Bundles any algo from the framework (in `../algos/`) into a Grand Challenge–ready Docker image. You pick the algo + config; the build script copies the framework into the build context, the Dockerfile installs deps and bakes the algo+config selection into env vars, and the save script gzips the image for upload.

## Quickstart

```bash
# 1. Place test data
mkdir -p test/input/interf0/images/stacked-neuron-images-with-noise
cp your_noisy_stack.tif test/input/interf0/images/stacked-neuron-images-with-noise/

# 2. Build for your chosen algo + config
./do_build.sh dvt_unet3d dvt_unet3d_t4

# 3. Test it locally (rebuilds if you pass new args)
./do_test_run.sh
# OR build + run in one shot:
./do_test_run.sh deepcad deepcad_default

# 4. When it works, save as gzipped tarball for Grand Challenge upload
./do_save.sh
```

The output ends up in `test/output/interf0/images/stacked-neuron-images-with-reduced-noise/`.

## Algo + config selection

Three ways to set which algo runs (highest precedence first):

1. **Per-run env var on `docker run`**: `-e SUBMISSION_ALGO=deepcad -e SUBMISSION_CONFIG=deepcad_rt` (this is what `do_test_run.sh` does behind the scenes)

2. **Build-arg** (`do_build.sh` arg1 arg2): bakes defaults into the image so `docker run` doesn't need any env vars. Grand Challenge runs your image without custom env vars, so **you must set the right defaults at build time** for your uploaded image.

3. **Edit `inference.py`** `DEFAULT_ALGO` / `DEFAULT_CONFIG_NAME` constants — these are the fallback if neither env var nor build-arg is set.

## What gets bundled

The Dockerfile copies:
- `algos/` — all 15 algorithm modules
- `configs/` — all default configs
- `runner/` — the framework's preprocessing, IO, eval helpers
- `inference.py` — the generic entrypoint
- `requirements.txt` — Python deps

So every image has every algo available; the env var picks which one runs. This means image size is the same whether you build for `dvt_unet3d` or `srdtrans`, but you get the flexibility to swap algos at run time without rebuilding.

## Optional pretrained weights

If you have a saved checkpoint, drop it at:

```
model/<algo>.pth
```

…where `<algo>` is the algo name (e.g. `model/deepcad.pth`). When the image runs, it looks for:

```
/opt/ml/model/<algo>.pth
/opt/ml/model/<algo>_weights.pth
/opt/ml/model/weights.pth
```

If any of those exist, **training is skipped** and the pretrained checkpoint is used directly for inference. Per-stack normalization params are recomputed from the input stack.

Grand Challenge users: upload the checkpoint as an optional model tarball. The framework expects it at `/opt/ml/model/`.

## File-by-file

| File | Purpose |
|---|---|
| `Dockerfile` | Image recipe. Has `ARG SUBMISSION_ALGO` / `SUBMISSION_CONFIG` build-args. |
| `do_build.sh [algo] [config]` | Stages framework into build context, runs `docker build` with the right args. |
| `do_test_run.sh [algo] [config]` | Runs the image on `test/input/` with GPU + `--network none`, mirroring Grand Challenge. |
| `do_save.sh [algo] [config]` | `docker save \| gzip` into a `.tar.gz` ready for upload. |
| `requirements.txt` | Python deps installed at build time (numpy, tifffile, SimpleITK, einops, scikit-image). |
| `inference.py` | Generic entrypoint that dynamically loads `algos.<name>` + `configs.<name>` and runs train+infer. |
| `model/` | Drop pretrained `.pth` here. Empty by default. |
| `.last_built_*` | Bookkeeping files written by `do_build.sh` so test/save can pick up the right tag. |

## Notes & gotchas

- **`--network none` in `do_test_run.sh`** mirrors what Grand Challenge does. If your algo unexpectedly tries to download something at run time (e.g. a model from HuggingFace), it will fail under `--network none`. Test locally first.

- **GPU**: `do_test_run.sh` passes `--gpus all`. Your local Docker needs nvidia-container-runtime configured. Without GPU, training will be extremely slow (and likely time out on Grand Challenge).

- **Output dtype matches input**: the inference script writes denoised stacks with the same dtype as input (e.g. int16 → int16 with clip+round).

- **Image tag** is derived from algo+config (e.g. `cidc25-submission-deepcad-deepcad_rt`) so multiple algos can coexist locally.

- **The framework's "training" varies per algo**: DeepCAD uses Noise2Noise (paper-faithful), SRDTrans uses spatial-redundancy sampling (paper-faithful), other algos use the framework's Noise2Void wrapper. Check each algo's docstring for what it does.
