# calo_classification

Classifier of particles from calorimeter simulation.

## Project Setup

```bash
git clone git@github.com:VictorAboba/calo_classification.git
cd calo_classification
uv init & uv sync --frozen
```

## Configuration

Edit `*_config.yaml` files to set your parameters:

| File | Description |
|------|-------------|
| `accelerate_config.yaml` | Accelerate framework configuration |
| `optuna_config.yaml` | Hyperparameters for Optuna optimization |
| `experiment_config.yaml` | Prefix for experiment folders/files, path to dataset splits |
| `optimizer_config.yaml` | AdamW optimizer parameters |
| `model_config.yaml` | Model parameters |

## Usage

```bash
# Run training
uv run accelerate launch train_and_test_model.py --batch_size 50 --num_epoch 50 --early_stop_epochs 2 --early_stop_threshold 0.01

# Hyperparameter optimization
uv run accelerate launch hyperparams_optimization.py
```
