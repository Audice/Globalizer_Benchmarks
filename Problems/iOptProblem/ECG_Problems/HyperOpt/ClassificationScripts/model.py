import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation='hswish'):
        padding = (kernel_size - 1) // 2
        act = nn.Hardswish() if activation == 'hswish' else nn.ReLU(inplace=True)
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_channels),
            act
        )

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, squeeze_factor=4):
        super().__init__()
        squeezed_channels = max(1, in_channels // squeeze_factor)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, squeezed_channels, 1)
        self.fc2 = nn.Conv1d(squeezed_channels, in_channels, 1)

    def forward(self, x):
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim))
        if use_se:
            layers.append(SqueezeExcite(hidden_dim))
        layers.append(nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        else:
            return out

class MobileNetV3Small1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=3, p=0.2, o_features=128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 16, 3, stride=2),
            InvertedResidual(16, 16, 3, stride=2, expand_ratio=1, use_se=True),
            InvertedResidual(16, 24, 3, stride=2, expand_ratio=1, use_se=False),
            InvertedResidual(24, 24, 3, stride=1, expand_ratio=1, use_se=False),
            InvertedResidual(24, 40, 5, stride=2, expand_ratio=1, use_se=True),
            InvertedResidual(40, 40, 5, stride=1, expand_ratio=1, use_se=True),
            ConvBNAct(40, 96, 1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(96, o_features),
            nn.Hardswish(),
            nn.Dropout(p),
            nn.Linear(o_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x