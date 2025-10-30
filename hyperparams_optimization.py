import multiprocessing as mp
import yaml
from math import ceil
import json
import os

from optuna import create_study, Trial, TrialPruned
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from vit import CustomViT
from data_routine import get_dataloader_from_path
from data_schemes import ViTConfig, OptimizerConfig
from model_processes import train, inference
from optuna_utils import create_trial_from_dict
from constants import OPTUNA_FILE, TARGET_SHAPE, NUM_CHANNELS, TRAIN_DATASET_LEN_80_360


def run_train_in_process(trial: Trial):
    optuna_config = yaml.load(open(OPTUNA_FILE, "r"), yaml.FullLoader)
    for k, v in optuna_config.items():
        if v["type"] == "float":
            v["min"] = float(v["min"])
            v["max"] = float(v["max"])
    trial = create_trial_from_dict(trial, optuna_config)

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
    model_config = ViTConfig.model_validate(trial_params)
    patch_size = trial_params.pop("patch_size")
    model_config.input_dim = patch_size**2 * NUM_CHANNELS
    model_config.h_split = int(TARGET_SHAPE[0] / patch_size)
    model_config.v_split = int(TARGET_SHAPE[1] / patch_size)
    model = CustomViT(model_config)

    optimizer_config = OptimizerConfig.model_validate(trial_params)
    optimizer = AdamW(model.parameters(), **optimizer_config.to_dict())

    total_train_steps = (
        ceil(TRAIN_DATASET_LEN_80_360 / trial_params["batch_size"])
        * trial_params["num_epoch"]
    )
    warmup_steps = int(total_train_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_training_steps=total_train_steps, num_warmup_steps=warmup_steps
    )

    train_dataloader = get_dataloader_from_path(
        "dataset/80_360_MeV_parquet/dataset_train.parquet",
        False,
        trial_params["batch_size"],
    )
    val_dataloader = get_dataloader_from_path(
        "dataset/80_360_MeV_parquet/dataset_val.parquet",
        False,
        trial_params["batch_size"],
    )
    test_dataloader = get_dataloader_from_path(
        "dataset/80_360_MeV_parquet/dataset_test.parquet",
        False,
        trial_params["batch_size"],
    )

    accelerator = Accelerator()
    try:
        train_result: dict = train(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            trial_params["num_epoch"],
            accelerator,
        )  # type: ignore
        test_result = inference(model, test_dataloader, accelerator)
        train_result["test"] = test_result

        os.makedirs("optuna_result", exist_ok=True)
        with open(f"optuna_result/{trial_num}_trial.json", "w") as file:
            json.dump(train_result, file)
        q.put(train_result["val"][-1]["loss"])
    except Exception as e:
        q.put((-1, e))
        raise Exception(e) from e


if __name__ == "__main__":
    import pickle

    mp.set_start_method("spawn")
    study = create_study(direction="minimize")
    study.optimize(run_train_in_process, 10, gc_after_trial=True)

    with open("optuna_study.pkl", "wb") as file:
        pickle.dump(study, file)
