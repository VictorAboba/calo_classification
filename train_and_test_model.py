import yaml
from math import ceil
import json
import argparse
import os

from torch.optim import AdamW
import torch
from transformers import get_cosine_schedule_with_warmup

from vit import CustomViT
from data_routine import get_dataloader_from_path
from data_schemes import ViTConfig, OptimizerConfig
from hardware import setup_hardware, build_accelerator
from model_processes import train, inference
from constants import (
    MODEL_CONFIG_FILE,
    EXPERIMENT_CONFIG_FILE,
    OPTIMIZER_CONFIG_FILE,
    TARGET_SHAPE,
    NUM_CHANNELS,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--early_stop_epochs", type=int, default=2)
    parser.add_argument("--early_stop_threshold", type=float, default=0.01)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    train_args = parse_args()

    experiment_config = yaml.load(open(EXPERIMENT_CONFIG_FILE, "r"), yaml.FullLoader)
    _, hw_profile = setup_hardware(experiment_config.get("hardware"))

    model_config = ViTConfig.model_validate(
        yaml.load(open(MODEL_CONFIG_FILE, "r"), yaml.FullLoader)
    )
    patch_size = int(TARGET_SHAPE[0] / model_config.h_split)
    model_config.input_dim = patch_size**2 * NUM_CHANNELS
    model = CustomViT(model_config)

    optimizer_config = OptimizerConfig.model_validate(
        yaml.load(open(OPTIMIZER_CONFIG_FILE, "r"), yaml.FullLoader)
    )
    optimizer_config.fused = hw_profile.fused_optimizer
    optimizer = AdamW(model.parameters(), **optimizer_config.to_dict())

    train_dataloader = get_dataloader_from_path(
        experiment_config["train_data"],
        patch_size,
        False,
        train_args.batch_size,
        **hw_profile.dataloader_kwargs(),
    )
    val_dataloader = get_dataloader_from_path(
        experiment_config["val_data"],
        patch_size,
        False,
        train_args.batch_size,
        **hw_profile.dataloader_kwargs(),
    )
    test_dataloader = get_dataloader_from_path(
        experiment_config["test_data"],
        patch_size,
        False,
        train_args.batch_size,
        **hw_profile.dataloader_kwargs(),
    )

    total_train_steps = (
        ceil(len(train_dataloader.dataset) / train_args.batch_size)
        * train_args.num_epoch
    )
    warmup_steps = int(total_train_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_training_steps=total_train_steps, num_warmup_steps=warmup_steps
    )

    accelerator = build_accelerator(hw_profile)
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
        early_stopping_threshold=train_args.early_stop_threshold,
        save_best=True,
    )  # type: ignore
    res = model.load_state_dict(
        torch.load(
            f"best_{experiment_config['prefix']}.pt", map_location=accelerator.device
        )
    )
    print(res)
    test_result = inference(model, test_dataloader, accelerator)
    train_result["test"] = test_result

    os.makedirs("models", exist_ok=True)

    with open(f"models/best_trial_{experiment_config['prefix']}.json", "w") as file:
        json.dump(train_result, file, indent=2)
