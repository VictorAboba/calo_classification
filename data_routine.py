from functools import partial

from einops import rearrange
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image
import cv2

from constants import TARGET_SHAPE


def images_to_patches(
    images: np.ndarray | torch.Tensor | list[Image.Image], patch_size: int
) -> torch.Tensor:
    if isinstance(images, list):
        image_arrays = []
        for image in images:
            assert isinstance(image, Image.Image), (
                f"images in the list must be PIL.Image.Image, but it is {type(image)}"
            )
            assert image.mode == "RGB", "images in the list must be in color mode (RGB)"
            image = np.array(image)
            image_arrays.append(image)
        images = np.stack(image_arrays)

    assert len(images.shape) == 4, (
        f"images must have shape BСHW, but have {images.shape=}"
    )
    B, С, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"image's height ({H}) and width ({W}) must be divisible by patch size {patch_size}"
    )

    images = rearrange(
        images,
        "b с (hn p1) (wn p2) -> b (hn wn) (с p1 p2)",
        p1=patch_size,
        p2=patch_size,
    )

    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    maxes = (
        torch.max(images.flatten(start_dim=1), dim=1, keepdim=True)
        .values.unsqueeze(-1)
        .expand_as(images)
    )  # find max value in each image by batch dimension
    images = images / maxes

    return images.to(torch.float32)


def create_tensor(layer, index_x, index_y, energy):
    """
    Для одного события возвращает массив float32 формы (2, 22, 96):
        images[0] : YZ-проекция (чётные слои, используется index_along_y)
        images[1] : XZ-проекция (нечётные слои, используется index_along_x)
    Номер слоя в проекции = исходный слой // 2 (0..21)
    """
    layer = np.asarray(layer)
    index_x = np.asarray(index_x)
    index_y = np.asarray(index_y)
    energy = np.asarray(energy)

    images = np.zeros((2, 22, 96), dtype=np.float32)

    # канал 0: YZ (чётные слои)
    mask_yz = layer % 2 == 0
    if mask_yz.any():
        proj_layer = layer[mask_yz] // 2
        np.add.at(images[0], (proj_layer, index_y[mask_yz]), energy[mask_yz])

    # канал 1: XZ (нечётные слои)
    mask_xz = layer % 2 != 0
    if mask_xz.any():
        proj_layer = layer[mask_xz] // 2
        np.add.at(images[1], (proj_layer, index_x[mask_xz]), energy[mask_xz])

    return images


def process_row(row):
    layer = row["layer"]
    index_x = row["index_along_x"]
    index_y = row["index_along_y"]
    energy = row["energy_release"]
    return create_tensor(layer, index_x, index_y, energy)


def collate_fn(batch: list[dict], patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    images_list = [process_row(row) for row in batch]
    images_list = [
        cv2.resize(image.transpose((1, 2, 0)), TARGET_SHAPE).transpose((2, 0, 1))
        for image in images_list
    ]
    images = np.stack(images_list)
    images = images_to_patches(images, patch_size)
    labels = torch.tensor([row["labels"] for row in batch], dtype=torch.long)
    return images, labels


def get_dataloader_from_path(
    path_to_parquet_file, patch_size, streaming, batch_size, num_workers=5
) -> DataLoader:
    dataset = load_dataset(
        "parquet", data_files=path_to_parquet_file, streaming=streaming
    )["train"]

    collate_fn_with_patch_size = partial(collate_fn, patch_size=patch_size)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_with_patch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return dataloader
