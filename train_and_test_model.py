import yaml
from math import ceil
import json
import argparse

from accelerate import Accelerator
from torch.optim import AdamW
import torch
from transformers import get_cosine_schedule_with_warmup

from vit import CustomViT
from data_routine import get_dataloader_from_path
from data_schemes import ViTConfig, OptimizerConfig
from model_processes import train, inference
from constants import (
    MODEL_CONFIG_FILE,
    OPTIMIZER_CONFIG_FILE,
    TARGET_SHAPE,
    TRAIN_DATASET_LEN_80_360,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--early_stop_epochs", type=int, default=2)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    train_args = parse_args()

    model_config = ViTConfig.model_validate(
        yaml.load(open(MODEL_CONFIG_FILE, "r"), yaml.FullLoader)
    )
    patch_size = int(TARGET_SHAPE[0] / model_config.h_split)
    model_config.input_dim = patch_size**2
    model = CustomViT(model_config)

    optimizer_config = OptimizerConfig.model_validate(
        yaml.load(open(OPTIMIZER_CONFIG_FILE, "r"), yaml.FullLoader)
    )
    optimizer = AdamW(model.parameters(), **optimizer_config.to_dict())

    total_train_steps = (
        ceil(TRAIN_DATASET_LEN_80_360 / train_args.batch_size) * train_args.num_epoch
    )
    warmup_steps = int(total_train_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_training_steps=total_train_steps, num_warmup_steps=warmup_steps
    )

    train_dataloader = get_dataloader_from_path(
        "dataset/80_360_MeV_parquet/dataset_train.parquet",
        False,
        train_args.batch_size,
    )
    val_dataloader = get_dataloader_from_path(
        "dataset/80_360_MeV_parquet/dataset_val.parquet",
        False,
        train_args.batch_size,
    )
    test_dataloader = get_dataloader_from_path(
        "dataset/80_360_MeV_parquet/dataset_test.parquet",
        False,
        train_args.batch_size,
    )

    accelerator = Accelerator()
    train_result: dict = train(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        train_args.num_epoch,
        accelerator,
        early_stopping=True,
        early_stopping_epochs=train_args.early_stop_epochs,
        save_best=True,
    )  # type: ignore
    res = model.load_state_dict(torch.load("best.pt", map_location=accelerator.device))
    print(res)
    test_result = inference(model, test_dataloader, accelerator)
    train_result["test"] = test_result

    with open(f"best_trial.json", "w") as file:
        json.dump(train_result, file)
