# models/custom/custom_model.py

import torch
import torch.nn as nn

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))  # 使用更常见的池化参数
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super(ResNet9, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)  # 输出尺寸：128 x 112 x 112 (假设输入 224x224)
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )

        self.conv3 = ConvBlock(128, 256, pool=True)  # 输出尺寸：256 x 56 x 56
        self.conv4 = ConvBlock(256, 512, pool=True)  # 输出尺寸：512 x 28 x 28
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),  # 输出尺寸：512 x 14 x 14
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, num_classes)  # 根据输入尺寸调整
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out