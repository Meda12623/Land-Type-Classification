import torch
import torch.nn as nn
from torchvision import models

def get_modified_googlenet(
    num_input_channels=13,
    num_output_classes=10,
    pretrained=True
):
    """
    Modify GoogLeNet to:
    - Accept custom number of input channels
    - Output custom number of classes
    - Disable auxiliary classifiers
    """

    print(f"Loading GoogLeNet (Pretrained={pretrained})...")

    model = models.googlenet(
        weights=models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None,
        aux_logits=True,
        transform_input=False
    )

    # =========================
    # Modify First Convolution
    # =========================
    original_conv1_module = model.conv1
    original_conv1 = original_conv1_module.conv

    new_conv = nn.Conv2d(
        in_channels=num_input_channels,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias
    )

    original_conv1_module.conv = new_conv

    nn.init.kaiming_normal_(
        original_conv1_module.conv.weight,
        mode='fan_out',
        nonlinearity='relu'
    )

    # =========================
    # Modify Classifier
    # =========================
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_output_classes)

    # =========================
    # Disable Auxiliary Heads
    # =========================
    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None

    print(
        f"Model adapted: Input={num_input_channels} channels | "
        f"Classes={num_output_classes}"
    )

    return model


