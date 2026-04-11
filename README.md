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
uv run train_and_test_model.py

# Hyperparameter optimization
uv run hyperparams_optimization.py
```
