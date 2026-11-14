# calo_classification

Classifier of particles (`antiproton`, `proton`, `pion+`) from a calorimeter simulation.

Every event is a set of hits (`layer`, `index_along_x`, `index_along_y`, `energy_release`).
Hits are binned into a 2-channel image: even layers give the YZ projection, odd layers the XZ
projection, `2 x 22 x 96` in total. The image is resized to `2 x 30 x 120`, cut into square
patches, each image is normalised by its maximum, and the patches go to a Vision Transformer
(`vit.py`: RMSNorm pre-norm blocks, QK-norm attention, SwiGLU FFN, CLS token, separable
2D positional embeddings).

Training uses [Accelerate](https://huggingface.co/docs/accelerate), hyperparameters are tuned with
[Optuna](https://optuna.org). Hardware is detected automatically and the training settings
(mixed precision, TF32, `torch.compile`, fused optimizer, DataLoader) are picked for it —
see [Hardware auto-detection](#hardware-auto-detection).

## Project structure

| File | Description |
|------|-------------|
| `train_and_test_model.py` | Train a model with the parameters from `model_config.yaml` / `optimizer_config.yaml`, then evaluate it on the test split |
| `hyperparams_optimization.py` | Optuna search over the space from `optuna_config.yaml`, every trial runs in its own subprocess |
| `hardware.py` | Hardware detection, hardware-dependent training profile, `accelerate_config.yaml` generator |
| `model_processes.py` | Training loop with early stopping, evaluation (loss, accuracy, precision, recall, F1, ROC AUC), prediction |
| `data_routine.py` | Hits → image → patches, DataLoader construction |
| `vit.py` | `CustomViT` model |
| `data_schemes.py` | Pydantic schemes of the configs |
| `constants.py` | Class names, image size, config file names |
| `optuna_utils.py` | Builds an Optuna trial from `optuna_config.yaml` |

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13 is installed by uv automatically).

```bash
git clone git@github.com:VictorAboba/calo_classification.git
cd calo_classification
uv sync --frozen
```

`uv sync` installs `torch` from PyPI. Which build that gives depends on the platform:

| Platform | What `uv sync` installs | What to do |
|----------|-------------------------|------------|
| Linux + NVIDIA | CUDA build + `triton` | nothing (needs a recent NVIDIA driver) |
| macOS (Apple Silicon) | build with MPS | nothing |
| Windows + NVIDIA | **CPU-only** build | install a CUDA build (below) |
| Linux + AMD (ROCm) | CUDA build, AMD GPU is not used | install a ROCm build (below) |
| Intel GPU (XPU) | no XPU support | install an XPU build (below) |

To replace the locked `torch` with a build for your accelerator, pick the index for your
hardware on [pytorch.org](https://pytorch.org/get-started/locally/) and reinstall `torch` from it:

```bash
# NVIDIA on Windows (cu128 supports GPUs up to Blackwell / RTX 50xx)
uv pip install --reinstall-package torch "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# AMD: --index-url https://download.pytorch.org/whl/rocm<version>
# Intel: --index-url https://download.pytorch.org/whl/xpu
```

After that run every command with `uv run --no-sync ...` (or activate `.venv`), otherwise
`uv run` may reinstall the locked build. Check the result with:

```bash
uv run --no-sync python hardware.py
```

If an NVIDIA/AMD GPU is present but the installed `torch` is a CPU-only build, `hardware.py`
prints a note about it.

## Data

The splits are parquet files, their paths are set in `experiment_config.yaml`
(`train_data`, `val_data`, `test_data`; by default `dataset/80_360_MeV_parquet/dataset_{train,val,test}.parquet`).
Every row is one event:

| Column | Type | Description |
|--------|------|-------------|
| `layer` | list[int] | calorimeter layer of each hit, `0..43` (even → YZ, odd → XZ) |
| `index_along_x` | list[int] | strip index along X, `0..95` |
| `index_along_y` | list[int] | strip index along Y, `0..95` |
| `energy_release` | list[float] | energy deposited in the hit |
| `labels` | int | `0` — antiproton, `1` — proton, `2` — pion+ |

## Hardware auto-detection

`hardware.py` detects the accelerator in this order: **NVIDIA CUDA → AMD ROCm → Intel XPU →
Apple MPS → CPU**, and both training scripts configure themselves from the result. At start they
print a report like this:

```
==================== Hardware ====================
backend         : cuda (nvidia)
device          : NVIDIA GeForce RTX 5070 Ti Laptop GPU x1, 11.9 GB, sm_120
cpu             : Intel Core Ultra 7 255HX, 20 threads
system          : Windows, torch 2.11.0+cu128
---------------- Training profile ----------------
mixed precision : bf16
tf32 matmul     : on
torch.compile   : off
fused AdamW     : on
dataloader      : num_workers=4, pin_memory=on
note            : torch.compile is off: triton is not installed
==================================================
```

What is chosen automatically:

| Hardware | Mixed precision | TF32 | `torch.compile` | DataLoader |
|----------|-----------------|------|-----------------|------------|
| NVIDIA Ampere and newer (sm ≥ 8.0) | `bf16` | on | on, if `triton` is installed | pinned memory, async copies |
| NVIDIA Volta / Turing (sm 7.x) | `fp16` (with grad scaling) | — | on, if `triton` is installed | pinned memory, async copies |
| NVIDIA Pascal and older | `no` | — | off | pinned memory, async copies |
| AMD MI100 / MI200 / MI300, RDNA3 / RDNA4 | `bf16` | — | on, if `triton` is installed | pinned memory, async copies |
| Other AMD GPUs (e.g. RDNA2) | `fp16` | — | on, if `triton` is installed | pinned memory, async copies |
| Intel XPU | `bf16` (or `fp16`) | — | on, if `triton` is installed | — |
| Apple MPS | `no` | — | off | — |
| CPU with AVX512-BF16 / AMX | `bf16` | — | off | fewer workers |
| Other CPU | `no` | — | off | fewer workers |

In all cases:

- **fused AdamW** is used wherever torch has a fused kernel (CUDA/ROCm, XPU, MPS, CPU);
- **DataLoader workers**: `min(4, CPU threads per process - 1)` on GPUs and at most 2 on CPU
  (the model itself needs the cores there); workers are persistent, so they are not re-spawned
  every epoch (expensive on Windows and macOS, where workers are started with `spawn`);
- **attention** goes through `torch.nn.functional.scaled_dot_product_attention`, so torch picks
  the fastest kernel available on the device (flash / memory-efficient / math).

`torch.compile` needs `triton`: it comes with `torch` on Linux (`pytorch-triton-rocm` with the ROCm
build); on Windows install the [`triton-windows`](https://github.com/woct0rdho/triton-windows)
release that matches your `torch` version.

### Overriding

Any automatic choice can be fixed in the `hardware` section of `experiment_config.yaml`:

```yaml
hardware:
  device: auto           # auto | cpu (force CPU even if a GPU is present)
  mixed_precision: auto  # auto | 'no' | fp16 | bf16
  compile: auto          # auto | true | false  (torch.compile with inductor)
  tf32: auto             # auto | true | false  (TF32 matmuls, NVIDIA Ampere+)
  num_workers: auto      # auto | <int>         (DataLoader workers per process)
```

These values take precedence over `mixed_precision` / `dynamo_config` from `accelerate_config.yaml`
and the `accelerate launch` command line.

### Multi-GPU

`accelerate_config.yaml` sets the process topology (how many processes / GPUs `accelerate launch`
starts). To generate it for the current machine (one process per GPU, NUMA affinity on
NVIDIA + Linux):

```bash
uv run python hardware.py --write-accelerate-config
```

## Configuration

Edit the `*_config.yaml` files to set your parameters:

| File | Description |
|------|-------------|
| `experiment_config.yaml` | Prefix for experiment folders/files, paths to dataset splits, Optuna settings (`n_trials`, early stopping), `hardware` section |
| `model_config.yaml` | Model parameters for `train_and_test_model.py` |
| `optimizer_config.yaml` | AdamW parameters (`lr`, `beta1`, `beta2`, `weight_decay`) |
| `optuna_config.yaml` | Hyperparameter search space (`int`, `float` with optional log scale, or a `list` of values) |
| `accelerate_config.yaml` | Accelerate launch configuration (process topology) |

Notes on `model_config.yaml`: the image is `30 x 120` (H x W), so the patch size must divide both
(5, 10, 15 or 30); `h_split = 120 / patch_size`, `v_split = 30 / patch_size`, `hid_dim` must be
divisible by `n_heads`. `input_dim` is recomputed as `2 * patch_size²`.

## Usage

### Training

```bash
uv run accelerate launch --config_file accelerate_config.yaml train_and_test_model.py --batch_size 50 --num_epoch 50 --early_stop_epochs 2 --early_stop_threshold 0.01
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 10 | batch size per process |
| `--num_epoch` | 10 | maximum number of epochs (cosine schedule with 5% warmup is built for it) |
| `--early_stop_epochs` | 2 | stop after this many epochs without improvement of the validation loss |
| `--early_stop_threshold` | 0.01 | minimal decrease of the validation loss that counts as an improvement |

The checkpoint with the best validation loss is saved to `best_<prefix>.pt`, the per-epoch
statistics and the test metrics to `models/best_trial_<prefix>.json`.

### Hyperparameter optimization

```bash
uv run accelerate launch --config_file accelerate_config.yaml hyperparams_optimization.py
```

Runs `optuna.n_trials` trials; every trial trains in a separate process (so GPU memory is fully
released between trials) with the early stopping settings from `experiment_config.yaml` and
minimises the validation loss. Results:

- `optuna_result_<prefix>/<n>_trial.json` — parameters, per-epoch statistics and test metrics of each trial;
- `optuna_result_<prefix>/best_trial.json` — the best trial;
- `optuna_study_<prefix>.pkl` — the pickled Optuna study.

To train the final model, copy the best parameters into `model_config.yaml` /
`optimizer_config.yaml` and run the training command.
