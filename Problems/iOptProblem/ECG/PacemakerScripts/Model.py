import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5,
                 dilation=1, dropout=0.3, use_batch_norm=True):
        super().__init__()

        self.conv1 = SeparableConv1D(in_channels, out_channels, kernel_size,
                                     padding=kernel_size // 2 * dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.act1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout * 0.5)

        self.conv2 = SeparableConv1D(out_channels, out_channels, kernel_size,
                                     padding=kernel_size // 2 * dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.act2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout * 0.5)

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
            )

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = x + residual
        return x


class Xception1D(nn.Module):
    def __init__(self,
                 input_channels=12,
                 num_blocks=3,
                 growth_rate=24,
                 kernel_size=5,
                 dropout=0.4,
                 use_batch_norm=True,
                 use_se=True):

        super().__init__()

        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.use_se = use_se

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_channels, growth_rate, kernel_size=7, padding=3),
            nn.BatchNorm1d(growth_rate) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = growth_rate

        for i in range(num_blocks):
            out_channels = growth_rate * (2 ** (i + 1))

            block = XceptionBlock(
                in_channels, out_channels,
                kernel_size=kernel_size,
                dilation=1,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
            self.blocks.append(block)

            if i < num_blocks - 1:
                self.pools.append(nn.MaxPool1d(2))
            else:
                self.pools.append(nn.Identity())

            in_channels = out_channels

            if use_se:
                se = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, in_channels // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels // 4, in_channels),
                    nn.Sigmoid()
                )
                setattr(self, f'se_{i}', se)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.squeeze(1)

        x = self.init_conv(x)

        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            x = block(x)

            if self.use_se:
                se = getattr(self, f'se_{i}', None)
                if se is not None:
                    x = x * se(x).unsqueeze(-1)

            x = pool(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)

        x = self.classifier(x)
        return x