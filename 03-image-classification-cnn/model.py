import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, use_res_connection=True) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False
        )
        self.b1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.b2 = nn.BatchNorm2d(num_features=out_channels)

        if use_res_connection:
            self.res_mapping = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.res_mapping = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.res_mapping = None

    def forward(self, x):
        x_res = self.res_mapping(x) if self.res_mapping is not None else None

        x = F.relu(self.b1(self.conv1(x)))
        x = self.b2(self.conv2(x))

        if x_res is not None:
            x += x_res

        x = F.relu(x)
        return x

class CNNImageClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, conv_channels=None, use_res_connection=True) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [16, 32, 64, 128]

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU()
        )

        blocks = []
        in_ch = conv_channels[0]

        for out_ch in conv_channels:
            stride = 1 if in_ch == out_ch else 2
            blocks.append(ConvBlock(in_channels=in_ch, out_channels=out_ch, stride=stride, use_res_connection=use_res_connection))
            blocks.append(ConvBlock(in_channels=out_ch, out_channels=out_ch, stride=1, use_res_connection=use_res_connection))
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*blocks)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
           nn.Linear(conv_channels[-1], 256),
           nn.ReLU(),
           nn.Dropout(p=0.5),
           nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.conv_blocks(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x
