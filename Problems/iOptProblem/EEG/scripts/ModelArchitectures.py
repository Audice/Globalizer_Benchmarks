import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
import warnings

class ConvBlock1D(nn.Module):
    """Базовый сверточный блок для 1D сигналов"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_bn: bool = True,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.3):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock1D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dropout_rate: float = 0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))

        x += residual
        x = self.activation(x)

        return x


class InceptionBlock1D(nn.Module):
    """Inception-подобный блок для 1D сигналов"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout_rate: float = 0.3):
        super().__init__()

        # Ветвь 1: 1x1 свертка (сжатие)
        self.branch1 = nn.Conv1d(in_channels, out_channels // 4, 1)

        # Ветвь 2: свертка с ядром 3
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, 1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_sizes[0], padding=kernel_sizes[0] // 2)
        )

        # Ветвь 3: свертка с ядром 5
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, 1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_sizes[1], padding=kernel_sizes[1] // 2)
        )

        # Ветвь 4: свертка с ядром 7
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, 1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_sizes[2], padding=kernel_sizes[2] // 2)
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class AttentionBlock1D(nn.Module):
    """Attention Block для 1D сигналов"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation блок для 1D"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class Simple1DCNN(nn.Module):
    """
    Простая 1D CNN
    Архитектура: 4 сверточных слоя с возрастающим числом фильтров
    Вход: (batch, channels, time)
    Выход: (batch, feature_dim)
    """

    def __init__(self,
                 input_channels: int = 23,
                 input_width: int = 2560,
                 feature_dim: int = 512,
                 dropout_rate: float = 0.3,
                 use_bn: bool = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.input_width = input_width

        # Сверточные блоки
        self.conv1 = ConvBlock1D(input_channels, 64, kernel_size=7, stride=2, use_bn=use_bn, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)

        self.conv2 = ConvBlock1D(64, 128, kernel_size=5, stride=2, use_bn=use_bn, dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool1d(3, stride=2, padding=1)

        self.conv3 = ConvBlock1D(128, 256, kernel_size=3, stride=2, use_bn=use_bn, dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool1d(3, stride=2, padding=1)

        self.conv4 = ConvBlock1D(256, 512, kernel_size=3, stride=2, use_bn=use_bn, dropout_rate=dropout_rate)

        # Adaptive pooling и классификатор
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, feature_dim)

        self._feature_dim_before_fc = 512

    def forward(self, x):
        # x: (batch, input_channels, input_length)
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_dim(self):
        return self.feature_dim


class Deep1DCNN(nn.Module):
    """
    Глубокая 1D CNN с большим количеством слоев
    Архитектура: 8 сверточных слоев с возрастающим числом фильтров
    """

    def __init__(self,
                 input_channels: int = 23,
                 input_width: int = 2560,
                 feature_dim: int = 512,
                 dropout_rate: float = 0.3,
                 use_bn: bool = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.input_width = input_width

        # Начальные слои
        self.conv1 = ConvBlock1D(input_channels, 32, kernel_size=7, stride=2, use_bn=use_bn)
        self.conv2 = ConvBlock1D(32, 64, kernel_size=5, stride=2, use_bn=use_bn)
        self.conv3 = ConvBlock1D(64, 128, kernel_size=3, stride=2, use_bn=use_bn)
        self.conv4 = ConvBlock1D(128, 256, kernel_size=3, stride=2, use_bn=use_bn)
        self.conv5 = ConvBlock1D(256, 256, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv6 = ConvBlock1D(256, 512, kernel_size=3, stride=2, use_bn=use_bn)
        self.conv7 = ConvBlock1D(512, 512, kernel_size=3, stride=1, use_bn=use_bn)
        self.conv8 = ConvBlock1D(512, 512, kernel_size=3, stride=1, use_bn=use_bn)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_dim(self):
        return self.feature_dim


class ResNet1D(nn.Module):
    """
    1D ResNet с настраиваемой глубиной

    Поддерживает: ResNet18, ResNet34, ResNet50
    """

    def __init__(self,
                 input_channels: int = 23,
                 input_width: int = 2560,
                 feature_dim: int = 512,
                 depth: int = 18,  # 18, 34, 50
                 dropout_rate: float = 0.3,
                 use_bn: bool = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.input_width = input_width
        self.depth = depth

        # Определяем количество блоков для каждого слоя
        if depth == 18:
            n_blocks = [2, 2, 2, 2]
        elif depth == 34:
            n_blocks = [3, 4, 6, 3]
        elif depth == 50:
            n_blocks = [3, 4, 6, 3]
        else:
            n_blocks = [2, 2, 2, 2]

        # Начальный слой
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64) if use_bn else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)

        # ResNet блоки
        self.layer1 = self._make_layer(64, 64, n_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, n_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, n_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, n_blocks[3], stride=2, dropout_rate=dropout_rate)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, feature_dim)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride, dropout_rate):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_dim(self):
        return self.feature_dim


class Inception1D(nn.Module):
    """
    1D Inception
    Параллельные свертки с разными размерами ядер
    """

    def __init__(self,
                 input_channels: int = 23,
                 input_width: int = 2560,
                 feature_dim: int = 512,
                 dropout_rate: float = 0.3,
                 use_bn: bool = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.input_width = input_width

        # Начальный слой
        self.conv1 = ConvBlock1D(input_channels, 64, kernel_size=7, stride=2, use_bn=use_bn)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)

        # Inception блоки
        self.inception1 = InceptionBlock1D(64, 256, kernel_sizes=[3, 5, 7], dropout_rate=dropout_rate)
        self.pool2 = nn.MaxPool1d(3, stride=2, padding=1)

        self.inception2 = InceptionBlock1D(256, 512, kernel_sizes=[3, 5, 7], dropout_rate=dropout_rate)
        self.pool3 = nn.MaxPool1d(3, stride=2, padding=1)

        self.inception3 = InceptionBlock1D(512, 512, kernel_sizes=[3, 5, 7], dropout_rate=dropout_rate)

        # Выход
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.inception1(x)
        x = self.pool2(x)

        x = self.inception2(x)
        x = self.pool3(x)

        x = self.inception3(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_dim(self):
        return self.feature_dim


class DenseNet1D(nn.Module):
    """
    1D DenseNet
    """

    def __init__(self,
                 input_channels: int = 23,
                 input_width: int = 2560,
                 feature_dim: int = 512,
                 growth_rate: int = 32,
                 n_layers: int = 4,
                 dropout_rate: float = 0.3,
                 use_bn: bool = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.input_width = input_width
        self.growth_rate = growth_rate
        self.n_layers = n_layers

        # Начальный слой
        self.conv1 = ConvBlock1D(input_channels, 64, kernel_size=7, stride=2, use_bn=use_bn)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)

        # Dense блоки
        self.dense1 = self._make_dense_block(64, growth_rate, n_layers, dropout_rate)
        self.trans1 = self._make_transition(64 + n_layers * growth_rate, 128)

        self.dense2 = self._make_dense_block(128, growth_rate, n_layers, dropout_rate)
        self.trans2 = self._make_transition(128 + n_layers * growth_rate, 256)

        self.dense3 = self._make_dense_block(256, growth_rate, n_layers, dropout_rate)
        self.trans3 = self._make_transition(256 + n_layers * growth_rate, 512)

        self.dense4 = self._make_dense_block(512, growth_rate, n_layers, dropout_rate)

        # Выход
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 + n_layers * growth_rate, feature_dim)

    def _make_dense_block(self, in_channels, growth_rate, n_layers, dropout_rate):
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer1D(in_channels + i * growth_rate, growth_rate, dropout_rate))
        return nn.Sequential(*layers)

    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.dense1(x)
        x = self.trans1(x)

        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)

        x = self.dense4(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_feature_dim(self):
        return self.feature_dim


class DenseLayer1D(nn.Module):
    """Слой для DenseNet"""

    def __init__(self, in_channels, growth_rate, dropout_rate=0.3):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, growth_rate * 4, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(growth_rate * 4)
        self.conv2 = nn.Conv1d(growth_rate * 4, growth_rate, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = F.relu(x1)
        x1 = self.conv1(x1)

        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.dropout(x1)

        return torch.cat([x, x1], dim=1)


class ModelFactory:
    """Фабрика для создания 1D CNN моделей"""

    AVAILABLE_MODELS = {
        'simple': Simple1DCNN,
        'deep': Deep1DCNN,
        'resnet18': ResNet1D,
        'resnet34': ResNet1D,
        'resnet50': ResNet1D,
        'inception': Inception1D,
        'densenet': DenseNet1D,
    }

    @classmethod
    def create_model(cls,
                     model_name: str,
                     input_channels: int = 23,
                     input_width: int = 2560,
                     feature_dim: int = 512,
                     dropout_rate: float = 0.3,
                     use_bn: bool = True,
                     **kwargs) -> nn.Module:
        """
        Создает 1D CNN модель по названию модлеи

        Args:
            model_name: имя модели
            input_channels: количество входных каналов
            input_width: длина сигнала
            feature_dim: размер выходного вектора признаков
            dropout_rate: вероятность dropout
            use_bn: использовать BatchNorm
            **kwargs: дополнительные параметры для модели
        """

        if model_name not in cls.AVAILABLE_MODELS:
            available = list(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        # Создаем модель с параметрами
        model_class = cls.AVAILABLE_MODELS[model_name]

        # Базовые параметры для всех моделей
        model_params = {
            'input_channels': input_channels,
            'input_width': input_width,
            'feature_dim': feature_dim,
            'dropout_rate': dropout_rate,
            'use_bn': use_bn,
        }

        # Особые параметры для разных моделей
        if model_name.startswith('resnet'):
            depth = int(model_name.replace('resnet', '')) if model_name != 'resnet' else 18
            model_params['depth'] = depth

        elif model_name == 'densenet':
            model_params['growth_rate'] = kwargs.get('growth_rate', 32)
            model_params['n_layers'] = kwargs.get('n_layers', 4)
            kwargs.pop('growth_rate', None)
            kwargs.pop('n_layers', None)

        model_params.update(kwargs)

        try:
            model = model_class(**model_params)
        except TypeError as e:
            print(f"Ошибка создания модели {model_name}: {e}")
            print(f"   Parameters: {model_params.keys()}")
            raise

        return model