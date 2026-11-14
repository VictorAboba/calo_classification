import multiprocessing as mp
import yaml
from math import ceil
import json
import os

from optuna import create_study, Trial, TrialPruned
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from vit import CustomViT
from data_routine import get_dataloader_from_path
from data_schemes import ViTConfig, OptimizerConfig
from hardware import setup_hardware, build_accelerator
from model_processes import train, inference
from optuna_utils import create_trial_from_dict
from constants import (
    OPTUNA_FILE,
    TARGET_SHAPE,
    NUM_CHANNELS,
    EXPERIMENT_CONFIG_FILE,
)


def run_train_in_process(trial: Trial):
    optuna_config = yaml.load(open(OPTUNA_FILE, "r"), yaml.FullLoader)
    for k, v in optuna_config.items():
        if v["type"] == "float":
            v["min"] = float(v["min"])
            v["max"] = float(v["max"])
    trial = create_trial_from_dict(trial, optuna_config)

    print(f"Trial {trial.number} with params: {json.dumps(trial.params, indent=2)}")

    mp_context = mp.get_context("spawn")
    q = mp_context.Queue()
    p = mp_context.Process(target=run_train, args=[trial.params, trial.number, q])
    p.start()
    p.join()

    if q.empty():
        raise TrialPruned("q is empty")
    else:
        result = q.get()
        if isinstance(result, tuple) and result[0] == -1:
            raise TrialPruned(result[1])

    return result


def run_train(trial_params: dict, trial_num, q):
    experiment_config = yaml.load(open(EXPERIMENT_CONFIG_FILE, "r"), yaml.FullLoader)
    # detection runs in the trial subprocess so the parent never initializes the GPU
    _, hw_profile = setup_hardware(experiment_config.get("hardware"))
    model_config = ViTConfig.model_validate(trial_params)
    patch_size = trial_params.pop("patch_size")
    model_config.input_dim = patch_size**2 * NUM_CHANNELS
    model_config.h_split = int(TARGET_SHAPE[0] / patch_size)
    model_config.v_split = int(TARGET_SHAPE[1] / patch_size)
    model = CustomViT(model_config)

    optimizer_config = OptimizerConfig.model_validate(trial_params)
    optimizer_config.fused = hw_profile.fused_optimizer
    optimizer = AdamW(model.parameters(), **optimizer_config.to_dict())

    train_dataloader = get_dataloader_from_path(
        experiment_config["train_data"],
        patch_size,
        False,
        trial_params["batch_size"],
        **hw_profile.dataloader_kwargs(),
    )
    val_dataloader = get_dataloader_from_path(
        experiment_config["val_data"],
        patch_size,
        False,
        trial_params["batch_size"],
        **hw_profile.dataloader_kwargs(),
    )
    test_dataloader = get_dataloader_from_path(
        experiment_config["test_data"],
        patch_size,
        False,
        trial_params["batch_size"],
        **hw_profile.dataloader_kwargs(),
    )

    total_train_steps = (
        ceil(len(train_dataloader.dataset) / trial_params["batch_size"])
        * trial_params["num_epoch"]
    )
    warmup_steps = int(total_train_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_training_steps=total_train_steps, num_warmup_steps=warmup_steps
    )

    accelerator = build_accelerator(hw_profile)
    try:
        train_result, best_val_loss = train(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            trial_params["num_epoch"],
            accelerator,
            early_stopping=experiment_config["optuna"]["early_stopping"]["enabled"],
            early_stopping_epochs=experiment_config["optuna"]["early_stopping"][
                "epochs"
            ],
            early_stopping_threshold=experiment_config["optuna"]["early_stopping"][
                "threshold"
            ],
        )
        test_result = inference(model, test_dataloader, accelerator)
        train_result["test"] = test_result

        os.makedirs(f"optuna_result_{experiment_config['prefix']}", exist_ok=True)
        with open(
            f"optuna_result_{experiment_config['prefix']}/{trial_num}_trial.json", "w"
        ) as file:
            to_save = {
                "trial_params": trial_params,
                "train_result": train_result,
                "best_val_loss": best_val_loss,
            }
            json.dump(to_save, file, indent=2)
        q.put(best_val_loss)
    except Exception as e:
        q.put((-1, e))
        raise Exception(e) from e


if __name__ == "__main__":
    import pickle

    mp.set_start_method("spawn")
    study = create_study(direction="minimize")

    experiment_config = yaml.load(open(EXPERIMENT_CONFIG_FILE, "r"), yaml.FullLoader)
    optuna_config = yaml.load(open(OPTUNA_FILE, "r"), yaml.FullLoader)
    print(f"Start optimization with config: {json.dumps(experiment_config, indent=2)}")
    print("*" * 20)
    print(f"Optuna config: {json.dumps(optuna_config, indent=2)}")
    print("*" * 20)

    study.optimize(
        run_train_in_process,
        experiment_config["optuna"]["n_trials"],
        gc_after_trial=experiment_config["optuna"]["gc_after_trial"],
    )

    os.makedirs(f"optuna_result_{experiment_config['prefix']}", exist_ok=True)
    with open(
        f"optuna_result_{experiment_config['prefix']}/best_trial.json", "w"
    ) as file:
        to_save = {
            "trial_params": study.best_trial.params,
            "trial_number": study.best_trial.number,
            "best_val_loss": study.best_trial.value,
        }
        json.dump(to_save, file, indent=2)

    with open(f"optuna_study_{experiment_config['prefix']}.pkl", "wb") as file:
        pickle.dump(study, file)
