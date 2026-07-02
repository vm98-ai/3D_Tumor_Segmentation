import logging
from typing import Tuple

import torch

log = logging.getLogger(__name__)

def convert_to_multichannel(label: torch.Tensor) -> torch.Tensor:
    """Map BraTS labels {1,2,3} -> 3 binary channels [WT, TC, ET]."""
    wt = (label == 1) | (label == 2) | (label == 3)
    tc = (label == 1) | (label == 3)
    et = (label == 3)
    return torch.cat([wt, tc, et], dim=0).float()


def build_transforms(roi_size: Tuple[int, int, int], train: bool):
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ConcatItemsd,
        DeleteItemsd, Spacingd, NormalizeIntensityd, RandSpatialCropd,
        RandFlipd, RandScaleIntensityd, RandShiftIntensityd, Lambdad,
    )

    keys_img = ["t2f", "t1n", "t1c", "t2w"]
    common = [
        LoadImaged(keys=keys_img + ["label"]),
        EnsureChannelFirstd(keys=keys_img + ["label"]),
        Orientationd(keys=keys_img + ["label"], axcodes="RAS"),
        ConcatItemsd(keys=keys_img, name="image", dim=0),
        DeleteItemsd(keys=keys_img),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Lambdad(keys="label", func=convert_to_multichannel),
    ]

    if train:
        return Compose(common + [
            RandSpatialCropd(keys=["image", "label"], roi_size=list(roi_size),
                             random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])
    return Compose(common)


def build_dataloaders(dataset_json: str, roi_size: Tuple[int, int, int],
                      batch_size: int, num_workers: int, val_fraction: float,
                      seed: int):
    import monai
    from monai.data import DataLoader, list_data_collate
    from monai.data import load_decathlon_datalist
    from monai.utils import set_determinism
    from sklearn.model_selection import train_test_split

    set_determinism(seed=seed)

    datalist = load_decathlon_datalist(dataset_json, True, "training")
    train_files, val_files = train_test_split(
        datalist, test_size=val_fraction, random_state=seed)
    log.info("Train: %d | Val: %d", len(train_files), len(val_files))

    train_ds = monai.data.Dataset(data=train_files, transform=build_transforms(roi_size, train=True))
    val_ds = monai.data.Dataset(data=val_files, transform=build_transforms(roi_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0,
                              collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(num_workers - 1, 0), pin_memory=True,
                            persistent_workers=num_workers > 1)
    return train_loader, val_loader, val_files
