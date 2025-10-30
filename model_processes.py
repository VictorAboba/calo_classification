import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np

from constants import CLASS_TO_NAME


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epoch: int,
    accelerator: Accelerator | None = None,
    early_stopping=False,
    early_stopping_epochs=5,
    save_best=False,
) -> dict | tuple[dict, float]:
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_upgrade = 0

    statistic = {"train": [], "val": []}

    if accelerator is not None:
        model, optimizer, train_dataloader, val_dataloader, scheduler = (
            accelerator.prepare(
                model, optimizer, train_dataloader, val_dataloader, scheduler
            )
        )
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, num_epoch + 1):
        print(f"{'='*50} Epoch {epoch} {'='*50}")

        train_loss_sum = 0
        pbar = tqdm(train_dataloader, desc="Train epoch")
        for i, batch in enumerate(pbar, 1):
            optimizer.zero_grad()

            inputs, labels = batch
            if accelerator is None:
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # -----------Backward step
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()
            # -------------------------
            optimizer.step()
            scheduler.step()

            if accelerator is not None:
                loss = accelerator.reduce(loss, reduction="mean")  # mean by devices
            train_loss_sum += loss.item()  # type: ignore

            pbar.set_postfix({"Avg_loss": round(train_loss_sum / i, 5)})
        pbar.close()

        val_metrics = inference(model, val_dataloader, accelerator)
        print(json.dumps(val_metrics, indent=4))

        statistic["train"].append(round(train_loss_sum / i, 5))
        statistic["val"].append(val_metrics)

        if early_stopping:
            if best_val_loss > val_metrics["loss"]:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_without_upgrade = 0
                if save_best:
                    if accelerator is not None:
                        accelerator.wait_for_everyone()
                        unwrap_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrap_model.state_dict(), "best.pt")
                    else:
                        torch.save(model.state_dict(), "best.pt")
            else:
                epochs_without_upgrade += 1
            if epochs_without_upgrade >= early_stopping_epochs:
                print(f"{'='*50} Early stopping on {epoch} epoch {'='*50}")
                print(f"Best epoch: {best_epoch}")
                return statistic, best_val_loss

    return statistic


def inference(
    model: nn.Module, dataloader: DataLoader, accelerator: Accelerator | None = None
) -> dict[str, float]:
    total_loss = 0
    total_predictions = []
    total_scores = []
    total_labels = []
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference")
        for batch in pbar:
            inputs, labels = batch
            if accelerator is None:
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                labels = labels.to(device)

            outputs = model(inputs)

            if accelerator is not None:
                outputs, labels = accelerator.gather_for_metrics((outputs, labels))
            predictions = torch.argmax(outputs, dim=-1).tolist()

            total_loss += loss_fn(outputs, labels).item()
            total_predictions.extend(predictions)
            total_labels.extend(labels.tolist())
            total_scores.extend(F.softmax(outputs, dim=-1).tolist())

            pbar.set_postfix(
                {"Avg_loss": round(total_loss / len(total_predictions), 5)}
            )
        pbar.close()

    avg_loss = round(total_loss / len(total_predictions), 5)
    acc = round(accuracy_score(total_labels, total_predictions), 5)
    recall = round(recall_score(total_labels, total_predictions, average="macro"), 5)  # type: ignore
    precision = round(precision_score(total_labels, total_predictions, average="macro"), 5)  # type: ignore
    f1 = round(f1_score(total_labels, total_predictions, average="macro"), 5)  # type: ignore
    roc_auc = round(
        roc_auc_score(total_labels, total_scores, average="macro", multi_class="ovr"), 5
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc,
    }  # type: ignore


def predict(
    model: nn.Module, dataloader: DataLoader, accelerator: Accelerator | None = None
) -> tuple[np.ndarray, list[str]]:
    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            if accelerator is None:
                device = next(model.parameters()).device
                inputs = inputs.to(device)
            outputs = model(inputs)

            classes = torch.argmax(outputs, dim=-1).detach()
            if accelerator is not None:
                classes = accelerator.gather_for_metrics(classes)
            predictions.append(classes)

    predictions = torch.cat(predictions).cpu().numpy()
    labels = [CLASS_TO_NAME[int(i)] for i in predictions]

    return predictions, labels
