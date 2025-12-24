import torch
import torch.nn as nn
from torchvision import models

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



class PrependResNet50(nn.Module):
    def __init__(self, num_classes=10, n_pre_convs=1, deep_head_dims=[1024,512], p_dropout=0.4, pretrained=True):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.base = base

        old_conv = self.base.conv1
        self.base.conv1 = nn.Conv2d(64, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                    stride=old_conv.stride, padding=old_conv.padding, bias=False)

        layers = []
        in_ch = 13
        out_ch = 64
        for i in range(n_pre_convs):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.prepend = nn.Sequential(*layers)

        self.prepend.apply(init_weights_kaiming)

        try:
            old_w = old_conv.weight.data
            pass
        except Exception:
            pass

        in_feat = self.base.fc.in_features
        head_layers = []
        last = in_feat
        for h in deep_head_dims:
            head_layers += [ nn.Linear(last, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(p_dropout) ]
            last = h
        head_layers += [ nn.Linear(last, num_classes) ]
        self.base.fc = nn.Sequential(*head_layers)
        self.base.fc.apply(init_weights_kaiming)

    def forward(self, x):
        x = self.prepend(x)    
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)
        return x

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)