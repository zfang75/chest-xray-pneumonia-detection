"""
Dataset and DataLoader utilities for the Chest X-Ray Pneumonia dataset.

The dataset is assumed to follow the standard Kaggle layout:
    data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/

Training DataLoader uses WeightedRandomSampler to balance the class distribution
(the training set contains roughly 3× more PNEUMONIA than NORMAL images).
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
_IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG")


def get_transforms(config: dict, split: str) -> transforms.Compose:
    """Build the image transform pipeline for a given data split.

    Training transforms include random augmentation; val/test transforms
    only resize and normalise.

    Args:
        config: Parsed YAML config dict.
        split:  One of 'train', 'val', or 'test'.

    Returns:
        A composed torchvision transform.
    """
    aug_cfg = config["augmentation"]
    norm_cfg = config["normalization"]
    image_size: int = norm_cfg["image_size"]
    mean: list = norm_cfg["mean"]
    std: list = norm_cfg["std"]

    normalize = transforms.Normalize(mean=mean, std=std)

    if split == "train":
        transform_list: list = [transforms.Resize((image_size, image_size))]

        if aug_cfg.get("random_horizontal_flip"):
            transform_list.append(transforms.RandomHorizontalFlip())

        rotation = aug_cfg.get("random_rotation", 0)
        affine_shear = aug_cfg.get("random_affine_shear", 0)
        if rotation or affine_shear:
            transform_list.append(
                transforms.RandomAffine(degrees=rotation, shear=affine_shear)
            )

        jitter = aug_cfg.get("color_jitter")
        if jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=jitter.get("brightness", 0),
                    contrast=jitter.get("contrast", 0),
                    saturation=jitter.get("saturation", 0),
                    hue=jitter.get("hue", 0),
                )
            )

        if aug_cfg.get("gaussian_blur"):
            # kernel size must be odd; 5% of image size works well for 224px input
            transform_list.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))

        transform_list += [transforms.ToTensor(), normalize]

        if aug_cfg.get("random_erasing"):
            # applied after ToTensor so it operates on the normalised tensor
            transform_list.append(
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
            )
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]

    return transforms.Compose(transform_list)


def _build_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that equalises class frequencies."""
    class_counts = [0] * len(CLASS_NAMES)
    for _, label in dataset.samples:
        class_counts[label] += 1

    class_weights = [1.0 / max(count, 1) for count in class_counts]
    sample_weights = [class_weights[label] for _, label in dataset.samples]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_data_loaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create and return train, val, and test DataLoaders.

    The training loader uses WeightedRandomSampler to address the class
    imbalance present in the chest X-ray dataset.  Val and test loaders
    iterate in a fixed order without shuffling.

    Args:
        config: Parsed YAML config dict.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_dir = Path(config["data"]["data_dir"])
    batch_size: int = config["training"]["batch_size"]
    num_workers: int = config["training"]["num_workers"]

    split_datasets = {
        split: datasets.ImageFolder(
            root=str(data_dir / split),
            transform=get_transforms(config, split),
        )
        for split in ("train", "val", "test")
    }

    sampler = _build_sampler(split_datasets["train"])

    train_loader = DataLoader(
        split_datasets["train"],
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        split_datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        split_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_class_distribution(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Count images per class for every split.

    Args:
        data_dir: Root directory of the dataset (contains train/val/test).

    Returns:
        Nested dict: {split: {class_name: count}}.
    """
    distribution: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        distribution[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = data_dir / split / class_name
            count = 0
            if class_dir.exists():
                for ext in _IMAGE_EXTENSIONS:
                    count += len(list(class_dir.glob(ext)))
            distribution[split][class_name] = count
    return distribution
