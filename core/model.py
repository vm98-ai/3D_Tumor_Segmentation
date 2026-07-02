import torch
from typing import Tuple

def build_model(in_channels: int = 4, num_classes: int = 3, dropout_p: float = 0.1):
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm

    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE,
        dropout=dropout_p,
    )


def build_loss(class_weights: Tuple[float, float, float] = (1.0, 1.0, 2.0)):
    from monai.losses import DiceCELoss
    weight = torch.tensor(class_weights, dtype=torch.float32)
    return DiceCELoss(sigmoid=True, include_background=True,
                      weight=weight, lambda_dice=1.0, lambda_ce=1.0)


def build_dice_metric():
    from monai.metrics import DiceMetric
    return DiceMetric(include_background=True, reduction="mean_batch")
