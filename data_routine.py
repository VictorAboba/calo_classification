from einops import rearrange
from datasets import load_dataset, IterableDataset, Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image


def images_to_patches(
    images: np.ndarray | torch.Tensor | list[Image.Image], patch_size: int
) -> torch.Tensor:
    if isinstance(images, list):
        image_arrays = []
        for image in images:
            assert isinstance(
                image, Image.Image
            ), f"images in the list must be PIL.Image.Image, but it is {type(image)}"
            assert (
                image.mode == "RGB"
            ), f"images in the list must be in color mode (RGB)"
            image = np.array(image)
            image_arrays.append(image)
        images = np.stack(image_arrays)

    assert (
        len(images.shape) == 4
    ), f"images must have shape BСHW, but have {images.shape=}"
    B, С, H, W = images.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), f"image's height ({H}) and width ({W}) must be divisible by patch size {patch_size}"

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


def collate_fn(batch: dict):  # -> dict[str, torch.Tensor]:
    pass


def get_dataloader_from_path(
    path_to_parquet_file, streaming, batch_size, num_workers=5
) -> DataLoader:
    dataset = load_dataset(
        "parquet", data_files=path_to_parquet_file, streaming=streaming
    )["train"]

    dataloader = DataLoader(
        dataset=dataset,  # type: ignore
        batch_size=batch_size,
        collate_fn=collate_fn,  # type: ignore
        shuffle=True,
        num_workers=num_workers,
    )

    return dataloader
