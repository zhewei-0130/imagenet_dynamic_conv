import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UniversalConvModule(nn.Module):
    def __init__(self, max_in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Weight generator: input channel mask => convolution kernels
        self.kernel_generator = nn.Sequential(
            nn.Linear(max_in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels * max_in_channels * kernel_size * kernel_size)
        )

    def forward(self, x, channel_mask):
        # x: [B, C, H, W], channel_mask: [B, C] or [C]
        B, C, H, W = x.size()
        # Expand mask to [B, C]
        if channel_mask.dim() == 1:
            channel_mask = channel_mask.unsqueeze(0).repeat(B, 1)
        # Generate weights per sample and use grouped convolution
        weights = self.kernel_generator(channel_mask)  # [B, out*C*k*k]
        weights = weights.view(B * self.out_channels, C, self.kernel_size, self.kernel_size)
        # Reshape input for grouped conv
        x = x.view(1, B * C, H, W)
        out = F.conv2d(x, weights, stride=self.stride, padding=self.padding, groups=B)
        # Restore batch dimension
        out_H, out_W = out.size(2), out.size(3)
        out = out.view(B, self.out_channels, out_H, out_W)
        return out

class ResNet18Dynamic(nn.Module):
    def __init__(self, num_classes=100, max_in_channels=3):
        super().__init__()
        base = models.resnet18(weights=None)
        self.dynamic_conv = UniversalConvModule(
            max_in_channels=max_in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        # reuse batchnorm, relu, pooling
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, channel_mask):
        x = self.dynamic_conv(x, channel_mask)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x